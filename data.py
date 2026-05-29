import PIL
import random as rng
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import to_pil_image

class ImagePairDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, transform: T.Transform | None = None):

        if split not in ['train', 'validate', 'test']:
            raise Exception('"split" parameter must be either "train" or "test"')

        if transform is None:
            transform = T.Compose([
                T.ToImage(), 
                T.ToDtype(torch.float32, scale=True)
            ])

        if split == 'test':
            self.data = CIFAR100(root='data', train=False, transform=transform)
            return

        self.data = CIFAR100(root='data', train=True, transform=transform)

        if split == 'train':
            self.data = Subset(self.data, range(40000))
        else:
            self.data = Subset(self.data, range(40000, 50000))


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        PIL_IMAGE = 0

        first_image = self.data[index][PIL_IMAGE]

        same = rng.random() < 0.5
        target = 1.0 if same else 0.0

        if same:
            second_index = index
        else:
            second_index = rng.randrange(0, len(self.data)-1)
            if second_index >= index:
                second_index += 1

        second_image = self.data[second_index][PIL_IMAGE]

        return first_image, second_image, target

    def __len__(self) -> int:
        return len(self.data)

if __name__ == '__main__':
    data = ImageDataset(split='train')
    first, second, target = data[0]
    print('Equal' if target == 1.0 else 'Different')
    to_pil_image(first).show()
    to_pil_image(second).show()