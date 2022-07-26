import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.append('../')
from data.data_loaders import load_scutfbp, load_hotornot, load_scutfbp5500_64, load_scutfbp5500_cv
from config.cfg import cfg
from util.file_util import mkdirs_if_not_exist
from pytorchcv.model_provider import get_model as ptcv_get_model


def main(data_name):
    """
        test load data

    """

    if data_name == 'SCUT-FBP':
        print('start loading SCUTFBPDataset...')
        dataloaders = load_scutfbp()
        # xent_weight_list = [91.5, 1.0, 1.06, 5.72, 18.3]
    elif data_name == 'HotOrNot':
        print('start loading HotOrNotDataset...')
        dataloaders = load_hotornot(cv_split_index=cfg['cv_index'])
        # xent_weight_list = [3.35, 1.0, 3.34]
    elif data_name == 'SCUT-FBP5500':
        print('start loading SCUTFBP5500Dataset...')
        dataloaders = load_scutfbp5500_64()
        # xent_weight_list = [1.88, 1, 1.91, 99.38, 99.38]
    elif data_name == 'SCUT-FBP5500-CV':
        print('start loading SCUTFBP5500DatasetCV...')
        dataloaders = load_scutfbp5500_cv(cv_index=cfg['cv_index'])
        # if cfg['cv_index'] == 1:
        #     xent_weight_list = [93.3, 1.98, 1.0, 1.91, 102.19]
        # elif cfg['cv_index'] == 2:
        #     xent_weight_list = [105.9, 1.92, 1.0, 1.86, 92.09]
        # elif cfg['cv_index'] == 3:
        #     xent_weight_list = [97.64, 1.97, 1.0, 1.92, 89.5]
        # elif cfg['cv_index'] == 4:
        #     xent_weight_list = [96.68, 1.92, 1.0, 1.9, 106.35]
        # elif cfg['cv_index'] == 5:
        #     xent_weight_list = [85.32, 1.94, 1.0, 1.9, 106.65]
    else:
        print('Invalid data name. It can only be [SCUT-FBP], [HotOrNot], [SCUT-FBP5500] or [SCUT-FBP5500-CV]...')
        sys.exit(0)
    for phase in ['train', 'val']:
        for i, data in enumerate(dataloaders[phase], 0):

            inputs = data['image']
            scores = data['score']
            classes = data['class']
            print( 'input shape {}'.format(inputs.shape))
            print('score {}'.format(scores))
            print('classes curr {}'.format(classes))

if __name__ == '__main__':
    seresnext50 = ptcv_get_model("seresnext50_32x4d", pretrained=True)
    num_ftrs = seresnext50.output.in_features
    seresnext50.output = nn.Linear(num_ftrs, 1)

    # resnet18 = models.resnet18(pretrained=True)
    # num_ftrs = resnet18.fc.in_features
    # resnet18.fc = nn.Linear(num_ftrs, 1)

    # main(ComboNet(num_out=5), 'SCUT-FBP', 'combinator')
    # main('SCUT-FBP5500')
    # main('SCUT-FBP')
    main('HotOrNot')
