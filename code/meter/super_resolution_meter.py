import numpy as np
import torch
import cv2
from torch import Tensor
from meter.utils import all_sum
from meter.utils import Meter


class SuperResolutionMeter(Meter):
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, phase: str, workers, scale):
        self._loss = []
        self._meters = [PSNR(), PSNR_GRAY()]
        self._records = {m.name: [] for m in self._meters}

        self.phase = phase
        self._scale = scale
        self._workers = workers

    def update(self, sr: Tensor, hr: Tensor, loss: Tensor):
        with torch.no_grad():
            sr = sr.data.cuda()
            hr = hr.data.cuda()
            loss = loss.data.cuda()

            if self._workers != 1:
                self._loss.append(all_sum(loss).cpu() / self._workers)
                for m in self._meters:
                    self._records[m.name].append(all_sum(m(sr, hr, self._scale)).cpu() / self._workers)
            else:
                self._loss.append(loss.cpu())
                for m in self._meters:
                    self._records[m.name].append(m(sr, hr, self._scale).cpu())

    def log_metric(self) -> dict:
        ret = {"loss": torch.mean(torch.stack(self._loss)), **{key: torch.mean(torch.stack(values)) for key, values in self._records.items()}}

        return {self.phase + "/" + key: value for key, value in ret.items()}

    def reset(self):
        self._loss = []
        self._records = {m.name: [] for m in self._meters}


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, scale):
        diff = (img1 - img2) / 255

        shave = scale + 6
        valid = diff[..., shave:-shave, shave:-shave]
        mse = valid.pow(2).mean() + 1e-10

        assert mse > 0

        return -10 * torch.log10(mse)

class PSNR_GRAY:
    def __init__(self):
        self.name = "PSNR_GRAY"

    @staticmethod
    def __call__(img1, img2, scale):
        diff = (img1 - img2) / 255

        shave = scale
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

        valid = diff[..., shave:-shave, shave:-shave]
        if valid.size == 0:
            return 0
        mse = valid.pow(2).mean() + 1e-10

        return -10 * torch.log10(mse)

class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return SSIM._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return SSIM._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()