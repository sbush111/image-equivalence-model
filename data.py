import numpy as np
import torch

class ImagePairDataset(torch.utils.data.Dataset):

    def __init__(self, split: str):
        if split not in ['train', 'validate', 'test']:
            raise Exception('"split" parameter must be either "train", "validate", or "test"')
        self.firsts = np.load(f'data/preprocessed/{split}_firsts.npy', mmap_mode='r')
        self.seconds = np.load(f'data/preprocessed/{split}_seconds.npy', mmap_mode='r')
        self.targets = np.load(f'data/preprocessed/{split}_targets.npy', mmap_mode='r')

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.tensor(self.firsts[index]), torch.tensor(self.seconds[index]), torch.tensor(self.targets[index])

    def __len__(self) -> int:
        return self.targets.shape[0]

if __name__ == '__main__':
    data = ImagePairDataset(split='train')
    first, second, target = data[2]
    from torchvision.transforms.functional import to_pil_image
    to_pil_image(first).show()
    to_pil_image(second).show()
    print(target)
    pass