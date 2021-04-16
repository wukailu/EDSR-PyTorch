from torch.utils.data import Dataset

from datasets.utils import FullDatasetBase
from torchvision.datasets import ImageFolder
from torchvision import transforms


class ImageNet(FullDatasetBase):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_shape = (3, 224, 224)
    num_classes = 1000
    name = "imagenet"

    def gen_train_transforms(self):
        base_transforms, _ = self.gen_base_transforms()
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            base_transforms
        ])
        return train_transforms, _

    def gen_test_transforms(self):
        base_transforms, _ = self.gen_base_transforms()
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            base_transforms
        ])
        return test_transforms, _

    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        return ImageFolder(root="/data/ImageNet/train",
                           transform=transform, target_transform=target_transform)

    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        return ImageFolder(root="/data/ImageNet/val",
                           transform=transform, target_transform=target_transform)

    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        return ImageFolder(root="/data/ImageNet/val",
                           transform=transform, target_transform=target_transform)

    @staticmethod
    def is_dataset_name(name: str):
        import re
        return re.match("(imagenet|ImageNet|Imagenet)$", name)
