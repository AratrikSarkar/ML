import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim  # dimension over which to apply softmax

    def forward(self, x):
        # Numerical stability: subtract max value along dim before exponentiating
        x_max, _ = torch.max(x, dim=self.dim, keepdim=True)
        x_exp = torch.exp(x - x_max)
        out = x_exp / torch.sum(x_exp, dim=self.dim, keepdim=True)
        return out

class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability must be between 0 and 1.")
        self.p = p  # probability of dropping a unit
        self.inplace = inplace

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        # Create a mask: 1 for keep, 0 for drop
        mask = (torch.rand_like(x) > self.p).float()

        if self.inplace:
            x.mul_(mask).div_(1 - self.p)  # scale to keep expected value
            out = x
        else:
            out = (x * mask) / (1 - self.p)

        return out

if __name__ == "__main__":
    torch.manual_seed(0)  # for reproducibility

    # ==== Test Softmax ====
    softmax = Softmax(dim=1)
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [0.5, 0.2, 0.3]])
    print("Input to Softmax:\n", x)
    print("Softmax output:\n", softmax(x))
    print("Sum along dim=1 (should be 1):\n", softmax(x).sum(dim=1))

    # ==== Test Dropout ====
    dropout = Dropout(p=0.3)
    dropout.train()  # enable dropout mode

    y = torch.ones((5, 5))
    print("\nInput to Dropout:\n", y)
    print("Dropout output (train mode):\n", dropout(y))

    dropout.eval()  # disable dropout
    print("\nDropout output (eval mode - should be unchanged):\n", dropout(y))