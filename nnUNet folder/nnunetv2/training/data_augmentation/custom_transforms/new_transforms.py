from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
import monai
import torchio as tio
from torchvision import transforms
import torch

class RandGibbsNoised(ImageOnlyTransform):
    def __init__(self):
        super().__init__()
        self.transforms = monai.transforms.RandGibbsNoised(keys=["images"], prob=0.1, alpha=(0.0, 1.0), allow_missing_keys=True)
        return

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self.transforms({'images': img})["images"]

class RandGaussianSmoothd(ImageOnlyTransform):
    def __init__(self):
        super().__init__()
        self.transforms = monai.transforms.RandGaussianSmoothd(keys=["images"], sigma_x=(1,2), sigma_y=(1,2), sigma_z=(1,2), approx='erf', prob=0.1, allow_missing_keys=True)
        return

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self.transforms({'images': img})["images"]
    
class RandScaleIntensityd(ImageOnlyTransform):
    def __init__(self):
        super().__init__()
        self.transforms = monai.transforms.RandScaleIntensityd(keys="images", factors=(0.7,1.3), prob=0.15, allow_missing_keys=True)
        return

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self.transforms({'images': img})["images"]
    
class RandGaussianNoised(ImageOnlyTransform):
    def __init__(self):
        super().__init__()
        self.transforms = monai.transforms.RandGaussianNoised(keys = ["images"], prob=0.1, mean=0.0, std=0.1, allow_missing_keys=True)
        return

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self.transforms({'images': img})["images"]

class RandomMotion(ImageOnlyTransform):
    def __init__(self):
        super().__init__()
        self.transforms = tio.transforms.RandomMotion(include=["images"], degrees=10, translation=10, num_transforms=1, image_interpolation='linear', p=0.1)
        return

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self.transforms({'images': img})["images"]

class RandomSpike(ImageOnlyTransform):
    def __init__(self):
        super().__init__()
        self.transforms = tio.transforms.RandomSpike(include=["images"], num_spikes=1, intensity=(1, 3), p=0.1)
        return

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self.transforms({'images': img})["images"]

class RandomBiasField(ImageOnlyTransform):
    def __init__(self):
        super().__init__()
        self.transforms = tio.transforms.RandomBiasField(include=["images"], coefficients=0.5, p=0.1)
        return

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self.transforms({'images': img})["images"]

class RandomElasticDeformation(ImageOnlyTransform):
    def __init__(self):
        super().__init__()
        self.transforms = tio.transforms.RandomElasticDeformation(include=["images"],num_control_points=7, max_displacement=7.5, p=0.1)
        return

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self.transforms({'images': img})["images"]

class RandomAnisotropy(ImageOnlyTransform):
    def __init__(self):
        super().__init__()
        self.transforms = tio.transforms.RandomAnisotropy(include=["images"],axes = (0,1,2), downsampling=(1, 2), image_interpolation='linear', p=0.1)
        return

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self.transforms({'images': img})["images"]
