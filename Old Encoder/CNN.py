import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        # x shape: (B, C_in, H_in, W_in)
        B, C_in, H_in, W_in = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        # Apply padding
        if pH > 0 or pW > 0:
            x = torch.nn.functional.pad(x, (pW, pW, pH, pH))

        H_out = (x.shape[2] - kH) // sH + 1
        W_out = (x.shape[3] - kW) // sW + 1

        out = torch.zeros((B, self.out_channels, H_out, W_out), device=x.device)

        # Perform convolution
        for b in range(B):
            for oc in range(self.out_channels):
                for i in range(0, H_out):
                    for j in range(0, W_out):
                        h_start, h_end = i * sH, i * sH + kH
                        w_start, w_end = j * sW, j * sW + kW
                        region = x[b, :, h_start:h_end, w_start:w_end]  # (C_in, kH, kW)
                        out[b, oc, i, j] = torch.sum(region * self.W[oc]) + (self.b[oc] if self.b is not None else 0)
        return out

class MaxPool2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride

        H_out = (H - kH) // sH + 1
        W_out = (W - kW) // sW + 1

        out = torch.zeros((B, C, H_out, W_out), device=x.device)

        for b in range(B):
            for c in range(C):
                for i in range(0, H_out):
                    for j in range(0, W_out):
                        h_start, h_end = i * sH, i * sH + kH
                        w_start, w_end = j * sW, j * sW + kW
                        region = x[b, c, h_start:h_end, w_start:w_end]
                        out[b, c, i, j] = torch.max(region)
        return out

if __name__ == "__main__":
    B, C_in, H, W = 2, 3, 5, 5  # batch=2, channels=3, height=5, width=5
    x = torch.randn(B, C_in, H, W).to(device)

    conv = Conv2D(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1).to(device)
    pool = MaxPool2D(kernel_size=2, stride=2).to(device)

    y = conv(x)
    print("Conv2D output shape:", y.shape)  # Expect: (B, out_channels, H, W)

    z = pool(y)
    print("MaxPool2D output shape:", z.shape)  # Expect: (B, out_channels, H/2, W/2)