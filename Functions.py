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

class Flatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        nd = x.dim()  # number of dimensions

        # Normalize possibly-negative dims
        start = self.start_dim if self.start_dim >= 0 else nd + self.start_dim
        end   = self.end_dim   if self.end_dim   >= 0 else nd + self.end_dim

        if not (0 <= start < nd) or not (0 <= end < nd):
            raise ValueError(f"start_dim/end_dim out of range for input with dim {nd}")
        if start > end:
            raise ValueError("start_dim must be <= end_dim")

        # Reshaping
        shape = list(x.shape)

        # If end is the last dim, don't append anything after -1
        if end == nd - 1:
            new_shape = shape[:start] + [-1]
        else:
            new_shape = shape[:start] + [-1] + shape[end + 1:]

        out = x.reshape(new_shape)
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

    # ==== Test Flatten ====
    flatten = Flatten()
    z = torch.randn(2, 3, 4, 5)  # shape: (2, 3, 4, 5)
    print("\nFlatten test:")
    print("Before flatten:", z.shape)
    z_flat = flatten(z)
    print("After flatten:", z_flat.shape)

    # Custom flatten range
    flatten_partial = Flatten(start_dim=2, end_dim=3)
    z_partial = flatten_partial(z)
    print("Partial flatten (dims 2-3):", z_partial.shape)