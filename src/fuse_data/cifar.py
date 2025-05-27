from torchvision.datasets import CIFAR10


class CIFAR10_mm(CIFAR10):

    def __init__(self, data_dir, is_train, preprocess_fn=None, tokenizer=None):
        super().__init__(
            root=data_dir,
            train=is_train,
            transform=preprocess_fn,
            download=True)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        img, label = super(CIFAR10_mm, self).__getitem__(idx)
        return img, self.tokenizer("")[0], label

    

