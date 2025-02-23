from abc import abstractmethod

import torch

from model.layerwise_model import pad_const_channel


def get_distill_module(name):
    methods = {
        'KD': KD,
        'CKA': CKA,
        'SRKD': SRKD,
        'FAKD': FAKD,
        'CKA_on_logits': CKA_on_logits,
        'KA_on_channel': KA_on_channel,
        'L2Distillation': L2Distillation,
        'L1Distillation': L1Distillation,
        'FD_Conv1x1': FD_Conv1x1,
        'FD_CloseForm': FD_CloseForm,
        'FD_BN1x1': FD_BN1x1,
        'FD_Conv1x1_MSE': FD_Conv1x1_MSE,
        'Progressive_FD': Progressive_FD,
    }
    return methods[name]


class DistillationMethod(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, feat_s, feat_t, epoch_ratio):
        pass


class BridgeDistill(DistillationMethod):
    def __init__(self, bridge, distill_loss='MSE', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge = bridge
        if distill_loss == 'MSE':
            self.loss = torch.nn.MSELoss()
        elif distill_loss == 'L1':
            self.loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError()

    def forward(self, feat_s, feat_t, epoch_ratio):
        assert len(feat_s) == len(self.bridge)
        loss = []
        for fs, ft, b in zip(feat_s, feat_t, self.bridge):
            loss.append(self.loss(b(pad_const_channel(fs)), ft))
        return torch.mean(torch.stack(loss))


class KD(DistillationMethod):
    def __init__(self, *args, T=3, **kwargs):
        super().__init__()
        self.T = T

    def forward(self, feat_s, feat_t, epoch_ratio):
        import torch.nn.functional as F
        loss = 0
        cnt = 0
        for fs, ft in zip(feat_s, feat_t):
            if len(fs.shape) == 2 and fs.shape == ft.shape:
                p_s = F.log_softmax(fs / self.T, dim=1)
                p_t = F.softmax(ft / self.T, dim=1)
                loss += F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / fs.size(0)
                cnt += 1
        return loss / cnt


class L2Distillation(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = []
        for fs, ft in zip(feat_s, feat_t):
            loss.append(torch.mean((fs - ft) ** 2))
        return torch.mean(torch.stack(loss))


class SRKD(DistillationMethod):
    def __init__(self, *args, eps=0.001, position=-1, **kwargs):
        super().__init__()
        self.eps = eps
        self.position = position

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = []
        for idx, (fs, ft) in enumerate(zip(feat_s, feat_t)):
            if self.position == -1 or idx in self.position:
                s = fs.mean() ** 2
                t = ft.mean() ** 2
                loss.append(((s - t) ** 2 + self.eps ** 2) ** 0.5)
        return torch.mean(torch.stack(loss))


class FAKD(DistillationMethod):
    def __init__(self, *args, position=-1, **kwargs):
        super().__init__()
        self.position = position

    @staticmethod
    def _spatial_similarity(fm):
        fm = fm.view(fm.size(0), fm.size(1), -1)
        norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
        s = norm_fm.transpose(1, 2).bmm(norm_fm)
        s = s.unsqueeze(1)
        return s

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = []
        for idx, (fs, ft) in enumerate(zip(feat_s, feat_t)):
            if self.position == -1 or idx in self.position:
                s = self._spatial_similarity(fs)
                t = self._spatial_similarity(ft)
                loss.append((t-s).abs().mean())
        return torch.mean(torch.stack(loss))


class L1Distillation(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = []
        for fs, ft in zip(feat_s, feat_t):
            loss.append(torch.mean(torch.abs(fs - ft)))
        return torch.mean(torch.stack(loss))


class FD_Conv1x1(DistillationMethod):
    def __init__(self, feat_s, feat_t, *args, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1) for fs, ft in zip(feat_s, feat_t) if
            len(fs.shape) == 4
        ])

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = []
        for fs, ft, conv in zip(feat_s, feat_t, self.convs):
            loss.append(torch.mean(torch.abs(conv(fs) - ft)))
        return torch.mean(torch.stack(loss))


class Progressive_FD(DistillationMethod):  # usually 5 for distill, 5*dist_loss + 1*cross_entropy
    def __init__(self, feat_s, feat_t, *args, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1) for fs, ft in zip(feat_s, feat_t) if
            len(fs.shape) == 4
        ])
        self.layer_idx = [idx for idx, (fs, ft) in enumerate(zip(feat_s, feat_t)) if len(fs.shape) == 4]

    def forward(self, feat_s, feat_t, epoch_ratio):
        assert 0 <= epoch_ratio <= 1
        loss = []
        idx = self.layer_idx[int((len(self.layer_idx) - 1e-4) * epoch_ratio)]
        fs, ft, conv = feat_s[idx], feat_t[idx], self.convs[idx]
        loss.append(torch.mean(torch.abs(conv(fs) - ft)))
        return torch.mean(torch.stack(loss))


class FD_Conv1x1_MSE(DistillationMethod):
    def __init__(self, feat_s, feat_t, *args, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1) for fs, ft in zip(feat_s, feat_t)
            if len(fs.shape) == 4 and fs.shape[2:] == ft.shape[2:]
        ])

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = []
        for fs, ft, conv in zip(feat_s, feat_t, self.convs):
            if len(fs.shape) == 4 and fs.shape[2:] == ft.shape[2:]:
                loss.append(torch.mean((conv(fs) - ft) ** 2))
        return torch.mean(torch.stack(loss))


class FD_BN1x1(DistillationMethod):
    def __init__(self, feat_s, feat_t, *args, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1) for fs, ft in zip(feat_s, feat_t)
        ])
        self.bn_t = torch.nn.ModuleList([
            torch.nn.BatchNorm2d(ft.size(1)) for fs, ft in zip(feat_s, feat_t)
        ])
        self.bn_s = torch.nn.ModuleList([
            torch.nn.BatchNorm2d(fs.size(1)) for fs, ft in zip(feat_s, feat_t)
        ])

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = 0
        for fs, ft, conv, bn_t, bn_s in zip(feat_s, feat_t, self.convs, self.bn_t, self.bn_s):
            ft = bn_t(ft)
            fs = bn_s(fs)
            loss += torch.mean(torch.abs(conv(fs) - ft))
        return loss


class FD_CloseForm(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = 0
        for fs, ft in zip(feat_s, feat_t):
            # 1x1 conv equivalent to AX = B, where A is of shape [N, HxW, C_s], X is [N, C_s, C_t], B is [N, HxW, C_t]
            # lstsq with batch is available in torch.linalg.lstsq with torch >= 1.9.0
            # torch.lstsq does not support grad
            A = torch.flatten(fs, start_dim=2).permute((0, 2, 1))
            B = torch.flatten(ft, start_dim=2).permute((0, 2, 1))

            # MSE over batches, this might be numerical unstable
            f_cnt = 0
            flag = False
            while not flag:
                flag = True
                A += torch.randn_like(A) * 0.1
                try:
                    s = A.pinverse() @ B
                except RuntimeError as e:
                    flag = False
                    f_cnt += 1
                if not flag:
                    A += torch.randn_like(A) * 0.1
            if f_cnt != 0:
                print(f"pinverse failed! repeated {f_cnt} times to get succeeded.")
                assert False

            r = (A @ s - B)
            loss += torch.mean(r ** 2)
        return loss


class CKA(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()
        from frameworks.nnmetric.feature_similarity_measurement import cka_loss
        self.cka = cka_loss()

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = 0
        cnt = 0
        for fs, ft in zip(feat_s, feat_t):
            loss += self.cka(fs, ft)
            cnt += 1
        return loss / cnt


class CKA_on_logits(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()
        from frameworks.nnmetric.feature_similarity_measurement import cka_loss
        self.cka = cka_loss()

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = 0
        cnt = 0
        for fs, ft in zip(feat_s, feat_t):
            if len(fs.shape) == 2 and fs.shape == ft.shape:
                loss += self.cka(fs, ft)
                cnt += 1
        return loss / cnt


class KA_on_channel(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()
        from frameworks.nnmetric.feature_similarity_measurement import ka_loss
        self.ka = ka_loss()

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = 0
        cnt = 0
        for fs, ft in zip(feat_s, feat_t):
            if len(fs.shape) == 4 and fs.shape[:3] == ft.shape[:3]:
                fs = fs.reshape(-1, fs.size(3))
                ft = ft.reshape(-1, ft.size(3))
            if len(fs.shape) == 2 and fs.shape[:1] == ft.shape[:1]:
                loss += self.ka(fs, ft)
                cnt += 1
        return loss / cnt
