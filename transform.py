import torch
from torchvision.transforms import v2 as T

def get_transform(to_tensor: bool = True,
                  rotation: int | float = 180,
                  translate: float = 0.2,
                  scale: float = 0.2,
                  flip: bool = True,
                  brightness: float = 0.2,
                  saturation: float = 0.2,
                  contrast: float = 0.2,
                  hue: float = 0.05,
                  blur: float | None = 0.1,
                  noise: float | None = 0.05) -> T.Transform:

    transforms = []

    if to_tensor is True:
        transforms.append(T.ToImage())
        transforms.append(T.ToDtype(torch.float32, scale=True))

    transforms.append(T.RandomAffine(degrees=rotation, 
                                     translate=(translate, translate),
                                     scale=(1 - scale, 1 + scale),
                                     interpolation=T.InterpolationMode.BILINEAR))

    if flip is True:
        transforms.append(T.RandomHorizontalFlip())

    transforms.append(T.ColorJitter(brightness=brightness,
                                    saturation=saturation,
                                    contrast=contrast,
                                    hue=hue))

    if blur is not None:
        transforms.append(T.GaussianBlur(kernel_size=5, sigma=(0.0001, blur)))

    if noise is not None:
        transforms.append(T.GaussianNoise(mean=0, sigma=noise))

    return T.Compose(transforms)

if __name__ == '__main__':
    print(get_transform())