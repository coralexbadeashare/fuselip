import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(
            self,
            hidden_size: int = 1536,
            input_size: int = 512,
            output_size: int = 2,
            device = 'cpu',
            ) -> None:
        super().__init__()
        
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.act = nn.functional.gelu
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.device = device
        self.to(device)

    def forward(self, x):
        # TODO: maybe add LN.
        out = self.lin1(x)
        out = self.act(out)
        out = self.lin2(out)
        return out


class ClassificationModel(nn.Module):
    def __init__(
        self,
        encoder,
        head,
        device='cpu',
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.device = device
        self.to(device)

    def forward(self, image=None, text=None):
        fts = self.encoder.encode_multimodal(image, text, normalize=True)
        logits = self.head(fts)
        return logits