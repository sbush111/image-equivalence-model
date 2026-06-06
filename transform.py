from config import Config
from torchvision.transforms import v2 as T
from typing import Optional
import torch

def generate_transform(config: Config) -> T.Transform:

    def jitter_to_range(jitter: Optional[float], base: float = 0.0) -> tuple[float, float]:
        if jitter is None:
            return None
        return (base - jitter, base + jitter)

    transforms = []

    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))

    transforms.append(T.RandomAffine(degrees=config.TRANSFORM_ROTATION or 0,
                                    translate=(config.TRANSFORM_TRANSLATE, config.TRANSFORM_TRANSLATE),
                                    scale=jitter_to_range(config.TRANSFORM_SCALE, base=1.0),
                                    interpolation=T.InterpolationMode.BILINEAR))

    if config.TRANSFORM_FLIP == True:
        transforms.append(T.RandomHorizontalFlip())

    transforms.append(T.ColorJitter(brightness=config.TRANSFORM_BRIGHTNESS,
                     contrast=config.TRANSFORM_CONTRAST,
                     saturation=config.TRANSFORM_SATURATION,
                     hue=config.TRANSFORM_HUE))

    if config.TRANSFORM_BLUR is not None:
        transforms.append(T.GaussianBlur(kernel_size=5, sigma=(0.0001, config.TRANSFORM_BLUR)))

    if config.TRANSFORM_NOISE is not None:
        transforms.append(T.GaussianNoise(mean=0, sigma=config.TRANSFORM_NOISE))

    return T.Compose(transforms)

if __name__ == '__main__':
    transform = generate_transform(Config())
    print(transform)