import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from model.AlexNet import AlexNet
from model.VGG16 import VGG_16_1D, VGG_19_1D
from model.ECSA_VGG16 import ECSA_VGG16, ECSA_VGG19
from model.ResNet import resnet152, resnet101, resnet34, resnet18, resnet50
from model.ECSA_ResNet import ecsa_resnet152, ecsa_resnet101, ecsa_resnet50, ecsa_resnet34, ecsa_resnet18, cgaps, cgmps, tgmps, tgaps, channelAttMaps, temporalAttMaps
from model.ECSA_ResNeXt import ecsa_resNeXt50, ecsa_resNeXt101
from model.ECSA_AlexNet import ECSA_AlexNet
from model.ResNeXt import resNeXt50, resNeXt101
from config import cfg
import pickle
import torch.nn.functional as F
from utils import read_sample_from_file, read_sample_from_h5, draw
from scipy.io import savemat
from signalUtils import readBinFile
from train import normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = {'AlexNet': AlexNet,
         'VGG': VGG_16_1D,
         'ResNet18': resnet18,
         'ResNet34': resnet34,
         'ResNet50': resnet50,
         'ResNet101': resnet101,
         'ResNet152': resnet152,
         'ResNeXt50': resNeXt50,
         'ResNeXt101': resNeXt101,
         'ECSA_AlexNet': ECSA_AlexNet,
         'ECSA_VGG16': ECSA_VGG16,
         'ECSA_ResNet18': ecsa_resnet18,
         'ECSA_ResNet34': ecsa_resnet34,
         'ECSA_ResNet50': ecsa_resnet50,
         'ECSA_ResNet101': ecsa_resnet101,
         'ECSA_ResNet152': ecsa_resnet152,
         'ECSA_ResNeXt50': ecsa_resNeXt50,
         'ECSA_ResNeXt101': ecsa_resNeXt101}


class SEI():
    def __init__(self, config):
        self.cfg = config
        self.sample_len = config['sample_len']
        self.sample_overlap = config['sample_overlap']
        self.load_model()

    def load_model(self):
        self.net = model[self.cfg['model']](self.cfg).to(device)
        data_file = self.cfg['h5_file'].split('/')[-1]
        data_file = data_file.split('.')[0]
        checkpoint_file = self.cfg['checkpoint_path'] + self.cfg['model'] + '/' + data_file + '_checkpoint_{:02d}.pth'.format(self.cfg['n_epoch'])
        checkpoint = torch.load(checkpoint_file)
        self.net.load_state_dict(checkpoint)

    def infer(self, IQs):
        IQs = torch.from_numpy(IQs)
        IQs = IQs.unsqueeze(0).to(device)
        IQs = normalize(IQs)
        out = self.net(IQs)
        out = F.softmax(out, dim=-1).cpu().detach().numpy()
        prob = out.max()
        index = out.argmax()
        return index, prob


if __name__ == '__main__':
    sei = SEI(cfg)
    ids = [0, 1, 2, 3, 4]
    for id in ids:
        print(id)
        I = readBinFile('data/USRPN2932/re_00_{}.bin'.format(id), 600)
        Q = readBinFile('data/USRPN2932/im_00_{}.bin'.format(id), 600)
        sample_count = min(len(I), len(Q))
        IQ = np.concatenate(([I[:sample_count]], [Q[:sample_count]]), axis=0)
        # sample, label = read_sample_from_h5(cfg['h5_file'], 'test', 0)
        for i in range(1000, 1100):
            sample = IQ[:, 600 * i:600 * i+512]
            index, prob = sei.infer(sample)
            print('index:{}, prob:{}'.format(index, prob))
            cgmps_list = [cgmps[i].squeeze().squeeze().cpu().detach().numpy() for i in range(len(cgmps))]
            cgaps_list = [cgaps[i].squeeze().squeeze().cpu().detach().numpy() for i in range(len(cgaps))]
            tgmps_list = [tgmps[i].squeeze().squeeze().cpu().detach().numpy() for i in range(len(tgmps))]
            tgaps_list = [tgaps[i].squeeze().squeeze().cpu().detach().numpy() for i in range(len(tgaps))]
            channelAttMaps_list = [channelAttMaps[i].squeeze().squeeze().cpu().detach().numpy() for i in
                                   range(len(channelAttMaps))]
            temporalAttMaps_list = [temporalAttMaps[i].squeeze().squeeze().cpu().detach().numpy() for i in
                                    range(len(temporalAttMaps))]
            savemat('data/tmp/{}_{}.mat'.format(id, i), mdict={'cgmps': cgmps_list,
                                                          'cgaps': cgaps_list,
                                                          'tgmps': tgmps_list,
                                                          'tgaps': tgaps_list,
                                                          'channelAttMaps': channelAttMaps_list,
                                                          'temporalAttMaps': temporalAttMaps_list})
            cgmps.clear()
            cgaps.clear()
            tgmps.clear()
            tgaps.clear()
            channelAttMaps.clear()
            temporalAttMaps.clear()
        print('---------------------------------------------')









