import os
from importlib import import_module

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.modules.loss._Loss):
    def __init__(self, hparams):
        super().__init__()
        print('Preparing loss function:')

        self.loss = []
        self.loss_module = nn.ModuleList()

        if isinstance(hparams['loss'], str):
            all_loss = [{'type': hparams['loss'], 'weight': 1}]
        elif isinstance(hparams['loss'], dict):
            all_loss = [hparams['loss']]
        elif isinstance(hparams['loss'], list) or isinstance(hparams['loss'], tuple):
            all_loss = [pair if isinstance(pair, dict) else {'type': pair, 'weight': 1} for pair in hparams['loss']]
        else:
            raise TypeError("loss must be a str or a dict of dicts")

        for loss in all_loss:
            weight, loss_type = loss['weight'], loss['type']
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                from frameworks.SuperResolution.loss.vgg import VGG
                loss_function = VGG(loss_type[3:], rgb_range=255)
            elif loss_type.find('GAN') >= 0:
                raise NotImplementedError("multiple optimizers need to change the framework")
                from frameworks.SuperResolution.loss.adversarial import Adversarial
                loss_function = Adversarial(gan_type=loss_type, **hparams)
            else:
                raise KeyError("loss type not recognized!")

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

            if loss_type.find('GAN') >= 0:
                raise NotImplementedError("multiple optimizers need to change the framework")
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
            #     self.log[-1, i] += effective_loss.item()
            # elif l['type'] == 'DIS':
            #     self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        # if len(self.loss) > 1:
        #     self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()
