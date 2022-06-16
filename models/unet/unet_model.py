""" Full assembly of the parts to form the complete network """
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
from tag.tag import Stage
from .unet_parts import *
from torch import nn
import torch
from .res_net import resnet34, resnet18, resnet50, resnet101, resnet152, BasicBlock, Bottleneck, ResNet
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


class SaveFeatures():
    features = None

    # def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    # def hook_fn(self, module, input, output): self.features = output

    # def remove(self): self.hook.remove()

    def __init__(self,m):
        self._outputs_lists = {}
        self.mymodule = m
        m.register_forward_hook(hook=self.save_output_hook)

    def save_output_hook(self, _, input, output):
        self._outputs_lists[input[0].device.index] = output
        self.features = self._outputs_lists

    def forward(self, x) -> list:
        self._outputs_lists[x.device.index] = []
        self.mymodule(x)
        return self._outputs_lists[x.device.index]


class UnetStageBlock(nn.Module):
    def __init__(self, stage, up_in, x_in, n_out, ratio):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.g_fc = nn.Linear(7,x_out * 2)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.stage = stage

        self.bn = nn.BatchNorm2d(n_out)

        self.pointwise = nn.Conv2d(14, n_out, kernel_size=1)
        self.depthwise = nn.Conv2d(n_out, n_out, kernel_size=3, stride=ratio , padding=1, groups=up_out) 

    def forward(self, up_p, x_p, give):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        res = self.bn(F.relu(cat_p))
        # g_p = self.g_fc(give).unsqueeze(-1).unsqueeze(-1) * res
        g_p = self.depthwise(self.pointwise(give))
        res = self.stage(g_p,res)
        return res

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        res = self.bn(F.relu(cat_p))
        return res

class TransUNet(nn.Module):

    def __init__(self, args, resnet='resnet34', num_classes=2, pretrained=False,
                in_chans=3,
                inplanes=64,
                num_layers=(3, 4, 6, 3),
                num_chs=(256, 512, 1024, 2048),
                num_strides=(1, 2, 2, 2),
                num_heads=(1, 2, 4, 8),
                num_parts=(1, 1, 1, 1),
                patch_sizes=(7, 7, 7, 8),
                drop_path=0.1,
                num_enc_heads=(1, 1, 1, 1),
                act=nn.GELU,
                ffn_exp=3,
                has_last_encoder=False
                ):
        super().__init__()
        # super(ResUnet, self).__init__()

        ''' ~~~~~ For the embedding transformer~~~~~'''
        cut, lr_cut = [8, 6]

        dim = args.dim #dim of transformer sequence, D of E

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
        '''~~~~~~End of embedding transformer~~~~~'''

        'unet and goinnet parameters'
        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')
        
        '''define the stage for goinnet giving'''
        last_chs = (256,256,256,256)
        num_chs = (256, 256, 256, 256)
        down_samples = (2,4,8,16)
        n_l = 1
        stage_list = []
        for i in range(4):
            stage_list.append(
                Stage(last_chs[i],
                    num_chs[i],
                    n_l,
                    num_heads=num_heads[i], #1,2,4,8
                    num_parts = (patch_sizes[i]**2 * (args.image_size // down_samples[i] // patch_sizes[i])**2),
                    patch_size=patch_sizes[i],  #8,8,8,8
                    drop_path=drop_path, #0.05
                    ffn_exp=ffn_exp,    #mlp hidden fea
                    last_enc=has_last_encoder and i == len(num_layers) - 1)
                        )
        self.stages = nn.ModuleList(stage_list)       
        '''end'''

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        self.check_layer = layers
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers


        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetStageBlock(self.stages[3], 512, 256, 256,16)
        self.up2 = UnetStageBlock(self.stages[2], 256, 128, 256,8)
        self.up3 = UnetStageBlock(self.stages[1], 256, 64, 256,4)
        self.up4 = UnetStageBlock(self.stages[0], 256, 64, 256,2)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.munet = MUNet(args)

        if args.sim_gpu:
            self.sim_gpu = args.sim_gpu
            self.gpu_device = args.gpu_device
        else:
            self.munet = MUNet(args)
            self.sim_gpu = 0
        '''~~~ self definition ~~~'''
        self.fc = nn.Linear(7,dim)
        self.tr_conv = nn.ConvTranspose2d(512, 512, 2, stride=2)

    def forward(self, x, cond, mod = 'train'):
        aux = {}
        mergf = []
        img = x
        # x = torch.cat((x,heatmap),1)
        x = F.relu(self.rn(x))              # x = [b_size, 2048, 8, 8]
        emb = x

        if mod == 'shuffle':
            self.up1.eval()
            self.up2.eval()
            self.up3.eval()
            self.up4.eval()
            self.up5.eval()

        '''~~~ 0: agg ~~~'''
        x = self.up1(x, self.sfs[3].features[x.device.index], cond)
        mergf.append(x)
        x = self.up2(x, self.sfs[2].features[x.device.index], cond)
        mergf.append(x)
        x = self.up3(x, self.sfs[1].features[x.device.index], cond)
        mergf.append(x)
        x = self.up4(x, self.sfs[0].features[x.device.index], cond)
        fea = x
        output = self.up5(x)
        '''end'''

        if mod == 'shuffle':
            aux['mergfs'] = mergf
            return output, aux

        if self.sim_gpu:
            self.munet = self.munet.to('cuda:' + str(self.sim_gpu))
            ave, mapsin = self.munet(img.to('cuda:' + str(self.sim_gpu)), output.detach().to('cuda:' + str(self.sim_gpu)))
            ave = ave.to('cuda:' + str(self.gpu_device))
            mapsin = mapsin.to('cuda:' + str(self.gpu_device))
            maps = [mapsin[:,i,:,:] for i in range(7)]
        else:
            ave, mapsin = self.munet(img, output.detach())
            maps = [mapsin[:,i,:,:] for i in range(7)]

        '''~~~ 0: ENDs ~~~'''

        pred_stack = torch.stack(maps) #7,b,c,w,w
        pred_stack_t = F.sigmoid(pred_stack)

        self_pred = pred_stack_t * torch.div(pred_stack_t, torch.sum(pred_stack_t, dim = 0, keepdim=True)) #7,b,c,w,w
        self_pred = rearrange(self_pred, "a b c h w -> b (a c) h w").contiguous() #b,7c,w,w
        cond = self_pred
        # maps = [nn.Upsample(scale_factor=2, mode='bilinear')(a) for a in maps]
        aux['maps'] = maps
        aux['cond'] = cond
        aux['mergfs'] = mergf
        aux['emb'] = emb
        return output, aux


    def close(self):
        for sf in self.sfs: sf.remove()

class MUNet(nn.Module):

    def __init__(self, args, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        # super(ResUnet, self).__init__()
        drop_path=0.05
        patch_sizes = (8,8,8,8)
        num_heads = (4,4,4,7)
        last_chs = (5,16,16,16)
        num_chs = (16, 16, 16, 14)
        down_samples = (4,4,4,4)
        n_l = 1
        stage_list = []
        for i in range(4):
            stage_list.append(
                Stage(last_chs[i],
                    num_chs[i],
                    n_l,
                    num_heads=num_heads[i], 
                    num_parts = (patch_sizes[i]**2 * (args.image_size // down_samples[i] // patch_sizes[i])**2),
                    patch_size=patch_sizes[i],  
                    drop_path=drop_path, 
                    ffn_exp=0.5,    
                    last_enc=0)
                        )
        self.stages = nn.ModuleList(stage_list)
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=4, padding=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.args = args

    def forward(self, x, heatmap):
        outlist = []
        x = torch.cat((x,heatmap),1)
        x = self.downsample(x)
        x = self.stages[0](x,x)
        # x = self.downsample(x)
        x = x.to('cuda:' + str(self.args.sim_gpu+2))
        stage1 = self.stages[1].to('cuda:' + str(self.args.sim_gpu+2))
        stage2 = self.stages[1].to('cuda:' + str(self.args.sim_gpu+2))
        x = stage1(x,x)
        x = stage2(x,x)
        x = x.to('cuda:' + str(self.args.gpu_device))
        # x = self.stages[1](x,x)
        # x = self.stages[2](x,x)
        # x = self.upsample(x)
        if self.args.sim_gpu:
            x = x.to('cuda:' + str(self.args.sim_gpu+1))
            last_stage = self.stages[3].to('cuda:' + str(self.args.sim_gpu+1))
            x = last_stage(x,x)
            x = x.to('cuda:' + str(self.args.gpu_device))
        # x = self.stages[3](x,x)
        x = self.upsample(x)
        x = rearrange(x, "b (g c) h w -> b g c h w", g = 7)
        outlist = [x[:,i,:,:] for i in range(7)]
        # outlist.append(x.item()[:,i,:,:] for i in range(7))


        return sum(outlist)/ 7, x

    def close(self):
        for sf in self.sfs: sf.remove()

class UNet(nn.Module):

    def __init__(self, args, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        # super(ResUnet, self).__init__()

        ''' ~~~~~ For the embedding transformer~~~~~'''
        cut, lr_cut = [8, 6]

        'unet and goinnet parameters'
        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained,inplanes = 3).children())[:cut]
        self.check_layer = layers
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers


        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

        self.pred1 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.pred2 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.pred3 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.pred4 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.pred5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.pred6 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.pred7 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.rn(x))              # x = [b_size, 2048, 8, 8]


        '''~~~ 0: Decoder ~~~'''
        x = self.up1(x, self.sfs[3].features[x.device.index])
        x = self.up2(x, self.sfs[2].features[x.device.index])
        x = self.up3(x, self.sfs[1].features[x.device.index])
        x = self.up4(x, self.sfs[0].features[x.device.index])
        fea = x
        output = self.up5(x)
        '''~~~ 0: ENDs ~~~'''
        '''
        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]
        '''
        return output

    def close(self):
        for sf in self.sfs: sf.remove()


