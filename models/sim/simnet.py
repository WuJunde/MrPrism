import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
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

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()



class Vgg(nn.Module):
    def __init__(self, num_classes=2):
        super(SimNet, self).__init__()
        self.n_class = num_classes
        # original image's size = 256*256*3

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_2 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2    2 layers

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4   2 layers

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_2 = nn.PReLU()
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8   4 layers

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_1 = nn.PReLU()
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_2 = nn.PReLU()
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.PReLU()
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.PReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.PReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.PReLU()       # 1/32    4 layers


        # ASPP module
        self.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.aspp1 = ASPP_module(512, 256, rate=1)
        self.aspp2 = ASPP_module(512, 256, rate=6)
        self.aspp3 = ASPP_module(512, 256, rate=12)
        self.aspp4 = ASPP_module(512, 256, rate=18)

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                             nn.Conv2d(512, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.PReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.PReLU())


        self.pred1 = nn.Conv2d(256, self.n_class, kernel_size=1, stride=1)
        self.pred2 = nn.Conv2d(256, self.n_class, kernel_size=1, stride=1)
        self.pred3 = nn.Conv2d(256, self.n_class, kernel_size=1, stride=1)
        self.pred4 = nn.Conv2d(256, self.n_class, kernel_size=1, stride=1)
        self.pred5 = nn.Conv2d(256, self.n_class, kernel_size=1, stride=1)
        self.pred6 = nn.Conv2d(256, self.n_class, kernel_size=1, stride=1)

        self.conv_c = nn.Conv2d(6, 512, 3, padding=1)
        self.CONVLSTMcell = ConvLSTMCell(512, 512)
        self.conv_f = nn.Conv2d(5, 3, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.zero_()
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


    def forward(self, x, coarse_pred,condition, flag=True):
        ''' Encoder '''
        y = coarse_pred
        x = torch.cat((x,y), dim=1)
        h = self.conv_f(x)
        o1= h

        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h = self.pool1(h)   #[b, 64, 128, 128]
        o2 = h

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h = self.pool2(h)   #[b, 128, 64, 64]
        o3 = h

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        low_level_fea = h   #[b, 256, 64, 64]
        h = self.pool3(h)   #[b, 256, 32, 32]
        o4 = h

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.pool4(h)   #[b, 512, 16, 16]
        o5 = h

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        o6 = h

        if flag:
            # add condition [bsize,6]
            condition = condition.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16)
            condition = self.conv_c(condition)

            for t in range(0, 1):                   # step =1
                state = self.CONVLSTMcell(h, [condition, condition])
            h = state[0]

            ''' Decoder '''
            x = self.up1(h)     #[b, 512, 32, 32]

            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x2 = x1+x2
            x3 = self.aspp3(x)
            x3 = x2+x3
            x4 = self.aspp4(x)
            x4 = x3+x4
            x5 = self.global_avg_pool(x)
            x5 = F.interpolate(x5, size=(32,32), mode='bilinear', align_corners=True)
            x5 = x4+x5

            x = torch.cat((x1, x2, x3, x4, x5), dim=1)
            x = self.relu(self.bn1(self.conv1(x)))      # [b, 256, 32, 32]
            x = F.interpolate(x, size=(64,64), mode='bilinear', align_corners=True)

            low_level_fea = self.relu(self.bn2(self.conv2(low_level_fea)))

            x = torch.cat((x, low_level_fea), dim=1)
            x = self.last_conv(x)
            x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=True)

            pred1 = self.pred1(x)
            pred2 = self.pred2(x)
            pred3 = self.pred3(x)
            pred4 = self.pred4(x)
            pred5 = self.pred5(x)
            pred6 = self.pred6(x)

            return [pred1,pred2,pred3,pred4,pred5,pred6]
        else:
            return [o1, o2, o3, o4, o5, o6]




    def copy_params_from_vgg16_bn(self, vgg16_bn):
        features = [
            self.conv1_1, self.bn1_1, self.relu1_1,
            self.conv1_2, self.bn1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.bn2_1, self.relu2_1,
            self.conv2_2, self.bn2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.bn3_1, self.relu3_1,
            self.conv3_2, self.bn3_2, self.relu3_2,
            self.conv3_3, self.bn3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.bn4_1, self.relu4_1,
            self.conv4_2, self.bn4_2, self.relu4_2,
            self.conv4_3, self.bn4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.bn5_1, self.relu5_1,
            self.conv5_2, self.bn5_2, self.relu5_2,
            self.conv5_3, self.bn5_3, self.relu5_3,
            # self.conv6,
            # self.conv7,
        ]
        for l1, l2 in zip(vgg16_bn.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            if isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=1)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

class SimNet(nn.Module):

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

        self.up12 = UnetBlock(512, 256, 256)
        self.up22 = UnetBlock(256, 128, 256)
        self.up32 = UnetBlock(256, 64, 256)
        self.up42 = UnetBlock(256, 64, 256)
        self.up52 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

        self.up13 = UnetBlock(512, 256, 256)
        self.up23 = UnetBlock(256, 128, 256)
        self.up33 = UnetBlock(256, 64, 256)
        self.up43 = UnetBlock(256, 64, 256)
        self.up53 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

        self.up14 = UnetBlock(512, 256, 256)
        self.up24 = UnetBlock(256, 128, 256)
        self.up34 = UnetBlock(256, 64, 256)
        self.up44 = UnetBlock(256, 64, 256)
        self.up54 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

        self.up15 = UnetBlock(512, 256, 256)
        self.up25 = UnetBlock(256, 128, 256)
        self.up35 = UnetBlock(256, 64, 256)
        self.up45 = UnetBlock(256, 64, 256)
        self.up55 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

        self.up16 = UnetBlock(512, 256, 256)
        self.up26 = UnetBlock(256, 128, 256)
        self.up36 = UnetBlock(256, 64, 256)
        self.up46 = UnetBlock(256, 64, 256)
        self.up56 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward(self, x):
        # x = torch.cat((x,piror),1)
        x = F.relu(self.rn(x))              # x = [b_size, 2048, 8, 8]

        # '''~~~ 1: Condition + feature ==> transformer ~~~'''
        # condition = self.fc(condition) #(b,dim)
        # x = self.to_patch_embedding(x) #b num_patch dim
        # b, n, _ = x.shape
        # # print('condition size is',condition.size())
        # # print('x size is',x.size())
        # x = torch.mul(x, condition.unsqueeze(1))
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.transformer(x)
        # x = rearrange(x, 'b (h w) d -> b d h w',h=4,w=4)
        # x = self.tr_conv(x)
        # # print('the size of x is',x.size())
        # '''~~~ 1: ENDs ~~~'''

        fea = x
        '''~~~ 0: Decoder ~~~'''
        x = self.up1(x, self.sfs[3].features[fea.device.index])
        x = self.up2(x, self.sfs[2].features[x.device.index])
        x = self.up3(x, self.sfs[1].features[x.device.index])
        x = self.up4(x, self.sfs[0].features[x.device.index])
        output = self.up5(x)
        '''~~~ 0: ENDs ~~~'''

        '''~~~ 0: Decoder ~~~'''
        x = self.up12(fea, self.sfs[3].features[fea.device.index])
        x = self.up22(x, self.sfs[2].features[x.device.index])
        x = self.up32(x, self.sfs[1].features[x.device.index])
        x = self.up42(x, self.sfs[0].features[x.device.index])
        output2 = self.up52(x)
        '''~~~ 0: ENDs ~~~'''

        '''~~~ 0: Decoder ~~~'''
        x = self.up13(fea, self.sfs[3].features[fea.device.index])
        x = self.up23(x, self.sfs[2].features[x.device.index])
        x = self.up33(x, self.sfs[1].features[x.device.index])
        x = self.up43(x, self.sfs[0].features[x.device.index])
        output3 = self.up53(x)
        '''~~~ 0: ENDs ~~~'''

        '''~~~ 0: Decoder ~~~'''
        x = self.up14(fea, self.sfs[3].features[fea.device.index])
        x = self.up24(x, self.sfs[2].features[x.device.index])
        x = self.up34(x, self.sfs[1].features[x.device.index])
        x = self.up44(x, self.sfs[0].features[x.device.index])
        output4 = self.up54(x)
        '''~~~ 0: ENDs ~~~'''

        '''~~~ 0: Decoder ~~~'''
        x = self.up15(fea, self.sfs[3].features[fea.device.index])
        x = self.up25(x, self.sfs[2].features[x.device.index])
        x = self.up35(x, self.sfs[1].features[x.device.index])
        x = self.up45(x, self.sfs[0].features[x.device.index])
        output5 = self.up55(x)
        '''~~~ 0: ENDs ~~~'''

        '''~~~ 0: Decoder ~~~'''
        x = self.up16(fea, self.sfs[3].features[fea.device.index])
        x = self.up26(x, self.sfs[2].features[x.device.index])
        x = self.up36(x, self.sfs[1].features[x.device.index])
        x = self.up46(x, self.sfs[0].features[x.device.index])
        output6 = self.up56(x)
        '''~~~ 0: ENDs ~~~'''

        '''
        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]
        '''
        return [output, output2, output3, output4, output5, output6] 

    def close(self):
        for sf in self.sfs: sf.remove()
