""" helper function

author baiyu
"""

import sys

import numpy

import torch
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import math
import PIL
import matplotlib.pyplot as plt
import seaborn as sns

import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz

from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from lucent.optvis.param.spatial import pixel_image, fft_image
from lucent.optvis.param.color import to_valid_rgb

import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

from collections import OrderedDict

from lucent.optvis import objectives, transform, param
from lucent.misc.io import show


def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """

    if net == 'vit':
        from vit_pytorch import ViT
        net = ViT(args)
    elif net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif net == 'transunet':
        from models.unet.unet_model import TransUNet
        model_cfg = dict(num_chs=(64, 64, 128, 256), patch_sizes=[8, 8, 8, 8], num_heads=(1, 2, 4, 8),
                      num_layers=[1, 1, 1, 1], ffn_exp=0.5, has_last_encoder=False, drop_path=0.05, pretrained = args.base_weights)
        args.model_cfg  = model_cfg
        net = TransUNet(args,**args.model_cfg)
    elif net == 'unet':
        from models.unet.unet_model import UNet
        net = UNet(args)
    elif net == 'sim':
        from models.sim.simnet import SimNet
        net = SimNet(args)
    elif net == 'efficient':
        from models.efficientnet import EfficientNet
        net = EfficientNet.from_name('efficientnet-b4',gpu_device)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        #net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net


def get_training_dataloader(args, batch_size=16, num_workers=2, shuffle=True):
    """ training primary backbone
    """
    data = args.train_data
    img_size = args.image_size
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        # transforms.RandomCrop((img_size, img_size)),  # padding=10
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_seg = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
    ])
    if data[0] == 'LAG':
        prob = [2/3, 1/3]  # probability of class 0 = 1/11, of 1 = 1/10

        Glaucoma_training = Dataset_DiscRegion(path,data,transform = transform_train, transform_seg = transform_seg)
        reciprocal_weights = []
        for index in range(len(Glaucoma_training)):
            _,_,label,_ = Glaucoma_training.__getitem__(index)
            reciprocal_weights.append(prob[label])

        weights = (1 / torch.Tensor(reciprocal_weights))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(Glaucoma_training))

        Glaucoma_training_loader = DataLoader(
            Glaucoma_training, num_workers=num_workers, batch_size=batch_size, sampler = sampler)

    elif data[0] == 'REFUGETrain' or data[0] == 'REFUGEVal'or data[0] == 'REFUGETest':
        prob = [10 / 11, 1 / 11]  # probability of class 0 = 1/11, of 1 = 1/10
        Glaucoma_training = Dataset_DiscRegion(path, data, transform=transform_train, transform_seg=transform_seg)
        reciprocal_weights = []
        for index in range(len(Glaucoma_training)):
            _, _, label, _ = Glaucoma_training.__getitem__(index)
            reciprocal_weights.append(prob[label])

        weights = (1 / torch.Tensor(reciprocal_weights))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(Glaucoma_training))

        Glaucoma_training_loader = DataLoader(
            Glaucoma_training, num_workers=num_workers, batch_size=batch_size, sampler = sampler)

    elif data[0] == 'Tongren1st' or data[0] == 'Tongren2nd':
        prob = [4/7, 3/7]  # probability of class 0 = 1/11, of 1 = 1/10

        Glaucoma_training = Dataset_FullImg(path,data,transform = transform_train, transform_seg = transform_seg)
        # reciprocal_weights = []
        # for index in range(len(Glaucoma_training)):
        #     _,_,label,_ = Glaucoma_training.__getitem__(index)
        #     reciprocal_weights.append(prob[label])

        # weights = (1 / torch.Tensor(reciprocal_weights))
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(Glaucoma_training))

        # Glaucoma_training_loader = DataLoader(
        #     Glaucoma_training, num_workers=num_workers, batch_size=batch_size, sampler = sampler)
        Glaucoma_training_loader = DataLoader(
            Glaucoma_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    else:
        Glaucoma_training = Dataset_FullImg(path, data, transform=transform_train, transform_seg=transform_seg)
        Glaucoma_training_loader = DataLoader(
            Glaucoma_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Glaucoma_training_loader

def get_valuation_dataloader(args, batch_size=16, num_workers=2, shuffle=True):
    """ valuationg data from the center of training
    """
    data = args.val_data
    img_size = args.image_size
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_seg = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


    Glaucoma_training = Dataset_FullImg(path,data,transform = transform_test, transform_seg = transform_seg)
    Glaucoma_training_loader = DataLoader(
        Glaucoma_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Glaucoma_training_loader

def get_test_dataloader(args, batch_size=16, num_workers=2, shuffle=True):
    """ test data
    """
    data = args.test_data
    img_size = args.image_size
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_seg = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


    Glaucoma_Test = Dataset_FullImg(path,data,transform = transform_test, transform_seg = transform_seg)
    Glaucoma_test_loader = DataLoader(
        Glaucoma_Test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Glaucoma_test_loader

def cka_loss(gram_featureA, gram_featureB):

    scaled_hsic = torch.dot(torch.flatten(gram_featureA),torch.flatten(gram_featureB))
    normalization_x = gram_featureA.norm()
    normalization_y = gram_featureB.norm()
    return scaled_hsic / (normalization_x * normalization_y)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0



@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    """
    Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    Returns:
        grid (Tensor): the tensor containing grid of images.
    Example:
        See this notebook
        `here <https://github.com/pytorch/vision/blob/master/examples/python/visualization_utils.ipynb>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


class RunningStats:
    def __init__(self, WIN_SIZE):
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE

        self.window = collections.deque(maxlen=WIN_SIZE)

    def clear(self):
        self.window.clear()
        self.mean = 0
        self.run_var = 0

    def is_full(self):
        return len(self.window) == self.WIN_SIZE

    def push(self, x):

        if len(self.window) == self.WIN_SIZE:
            # Adjusting variance
            x_removed = self.window.popleft()
            self.window.append(x)
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)
        else:
            # Calculating first variance
            self.window.append(x)
            delta = x - self.mean
            self.mean += delta / len(self.window)
            self.run_var += delta * (x - self.mean)

    def get_mean(self):
        return self.mean if len(self.window) else 0.0

    def get_var(self):
        return self.run_var / len(self.window) if len(self.window) > 1 else 0.0

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.window)

    def __str__(self):
        return "Current window values: {}".format(list(self.window))

def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

'''parameter'''
def para_image(w, h=None, img = None, seg = None, sd=None, batch=None,
          fft = False, channels=None):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    params, maps_f = param_f(shape, sd=sd)
    if seg:
        output = to_valid_out(maps_f,img,seg)
    else:
        output = gene_out(maps_f,img)
    return params, output

def to_valid_out(maps_f,img,seg):
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        maps = torch.nn.Softmax(dim = 1)(maps)
        final_seg = torch.multiply(seg,maps).sum(dim = 1, keepdim = True)
        return torch.cat((img,final_seg),1)
        # return torch.cat((img,maps),1)
    return inner

def gene_out(maps_f,img):
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        return maps * 0.15 + img
        # return torch.cat((img,maps),1)
    return inner


class CompositeActivation(torch.nn.Module):

    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1)


def cppn(size, img = None, seg = None, batch=None, num_output_channels=3, num_hidden_channels=24, num_layers=8,
         activation_fn=CompositeActivation, normalize=False, device = "cuda:0"):

    r = 3 ** 0.5

    coord_range = torch.linspace(-r, r, size)
    x = coord_range.view(-1, 1).repeat(1, coord_range.size(0))
    y = coord_range.view(1, -1).repeat(coord_range.size(0), 1)

    input_tensor = torch.stack([x, y], dim=0).unsqueeze(0).repeat(batch,1,1,1).to(device)

    layers = []
    kernel_size = 1
    for i in range(num_layers):
        out_c = num_hidden_channels
        in_c = out_c * 2 # * 2 for composite activation
        if i == 0:
            in_c = 2
        if i == num_layers - 1:
            out_c = num_output_channels
        layers.append(('conv{}'.format(i), torch.nn.Conv2d(in_c, out_c, kernel_size)))
        if normalize:
            layers.append(('norm{}'.format(i), torch.nn.InstanceNorm2d(out_c)))
        if i < num_layers - 1:
            layers.append(('actv{}'.format(i), activation_fn()))
        else:
            layers.append(('output', torch.nn.Sigmoid()))

    # Initialize model
    net = torch.nn.Sequential(OrderedDict(layers)).to(device)
    # Initialize weights
    def weights_init(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, 0, np.sqrt(1/module.in_channels))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    net.apply(weights_init)
    # Set last conv2d layer's weights to 0
    torch.nn.init.zeros_(dict(net.named_children())['conv{}'.format(num_layers - 1)].weight)
    return net.parameters(), gene_out(lambda: net(input_tensor),img)

'''adversary'''
def render_vis(
    args,
    model,
    objective_f,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(256,),
    verbose=True,
    preprocess=True,
    progress=True,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
    fixed_image_size=None,
    label = 1
):
    if label == 1:
        sign = 1
    elif label == 0:
        sign = -1
    else:
        print('label is wrong, label is',label)

    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()
    
    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = []
    transforms = transforms.copy()

    # Upsample images smaller than 224
    image_shape = image_f().shape

    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    hook = hook_model(model, image_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss of ad: {:.3f}".format(objective_f(hook)))

    images = []
    try:
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            def closure():
                optimizer.zero_grad()
                try:
                    model(transform_f(image_f()))
                except RuntimeError as ex:
                    if i == 1:
                        # Only display the warning message
                        # on the first iteration, no need to do that
                        # every iteration
                        warnings.warn(
                            "Some layers could not be computed because the size of the "
                            "image is not big enough. It is fine, as long as the non"
                            "computed layers are not used in the objective function"
                            f"(exception details: '{ex}')"
                        )
                if args.reverse:
                    loss = -sign * objective_f(hook)
                else:
                    loss = sign * objective_f(hook)
                loss.backward()
                return loss
                
            optimizer.step(closure)
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                if verbose:
                    print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                    if show_inline:
                        show(image)
                images.append(image)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))

    if save_image:
        image_name = image_name[0].split('\\')[-1].split('.')[0] + '.png'
        img_path = os.path.join(args.path_helper['sample_path'], str(image_name))
        export(image_f(), img_path)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    return image_f()


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tensor.size(1)
    # if c == 7:
    #     for i in range(c):
    #         w_map = tensor[:,i,:,:].unsqueeze(1)
    #         w_map = tensor_to_img_array(w_map).squeeze()
    #         w_map = (w_map * 255).astype(np.uint8)
    #         image_name = image_name[0].split('/')[-1].split('.')[0] + str(i)+ '.png'
    #         wheat = sns.heatmap(w_map,cmap='coolwarm')
    #         figure = wheat.get_figure()    
    #         figure.savefig ('./fft_maps/weightheatmap/'+str(image_name), dpi=400)
    #         figure = 0
    # else:
    if c == 3:
        vutils.save_image(tensor, fp = img_path)
    else:
        image = tensor[:,0:3,:,:]
        w_map = tensor[:,-1,:,:].unsqueeze(1)
        image = tensor_to_img_array(image)
        w_map = 1 - tensor_to_img_array(w_map).squeeze()
        # w_map[w_map==1] = 0
        assert len(image.shape) in [
            3,
            4,
        ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
        # Change dtype for PIL.Image
        image = (image * 255).astype(np.uint8)
        w_map = (w_map * 255).astype(np.uint8)

        Image.fromarray(w_map,'L').save(img_path)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None


    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output


    def close(self):
        self.hook.remove()


def hook_model(model, image_f):
    features = OrderedDict()
    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook

def vis_image(imgs, pred_masks, gt_masks, save_path, reverse = False):
    '''
    vis the segmentaion results
    imgs: a tensor with size [b,3,h,w]
    masks: [b,2,h,w], disk/cup: a tensor with size [b, 1, h, w]
    '''
    b,c,h,w = pred_masks.size()
    row_num = min(b, 4)
    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
    gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
    tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
    # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
    compose = torch.cat((pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
    vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)

    return

def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
    for th in threshold:

        gt_vmask_p = (true_mask_p > th).float()
        vpred = (pred > th).float()
        vpred_cpu = vpred.cpu()
        disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
        cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

        disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
        cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
   
        '''iou for numpy'''
        iou_d += iou(disc_pred,disc_mask)
        iou_c += iou(cup_pred,cup_mask)

        '''dice for torch'''
        disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
        cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
        
    return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)


















