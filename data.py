import PIL
import random as rng
import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import to_pil_image

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, transform: T):

        if split not in ['train', 'test']:
            raise Exception('"split" parameter must be either "train" or "test"')

        self.data = CIFAR100(root='data', train=(split=='train'), transform=transform)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        PIL_IMAGE = 0

        first_image = self.data[index][PIL_IMAGE]

        same = rng.random() < 0.5
        target = 1.0 if same else 0.0

        if same:
            second_index = index
        else:
            second_index = rng.randint(0, len(self.data)-1)
            if second_index >= index:
                second_index += 1

        second_image = self.data[second_index][PIL_IMAGE]

        return first_image, second_image, target

    def __len__(self) -> int:
        return len(self.data)

transform = T.Compose([
    T.ToTensor()
])

data = ImageDataset(split='train', transform=transform)

first, second, target = data[0]

first = to_pil_image(first)
second = to_pil_image(second)

first.show()
second.show()
print(target)