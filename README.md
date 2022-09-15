# Multi-rater Prism
A pytorch implementation of the paper 'Multi-rater Prism: Learning self-calibrated medical image segmentation from multiple raters' and 'Learning self-calibrated optic disc and cup segmentation from multi-rater annotations' accepted by MICCAI 2022

<p align="center"><img src="https://github.com/WuJunde/MrPrism/blob/main/prism.jpg" alt="text" width="400"/></p>

## Preparation

The code is run on pytorch1.8.1 + cuda 10.1.

## Quick Start

#### Training:

python train.py -net transunet -exp_name test_train -mod rec

#### Inference:

python val.py -net transunet -mod rec -exp_name val_seg -weights 'recorded weights'

See cfg.py for more avaliable parameters



### Todo list

- [ ] add requirement
- [x] del debug code
- [ ] function name alignment
- [ ] del redundance
- [ ] release a slim version

