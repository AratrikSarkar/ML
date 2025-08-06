import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.randn((fan_out, fan_in)))
        self.b = nn.Parameter(torch.randn(fan_out)) if bias else None

    def forward(self, x):
        out = x @ self.W.T
        if self.b is not None:
            out += self.b
        return out

class Sigmoid(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        positive = x >= 0
        negative = ~positive
        out[positive] = 1 / (1 + torch.exp(-x[positive]))
        exp_x = torch.exp(x[negative])
        out[negative] = exp_x / (1 + exp_x)
        return out

class TanH(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        positive = x >= 0
        negative = ~positive
        out[positive] = 1 - (2 / (torch.exp(2 * x[positive]) + 1))
        out[negative] = (2 / (torch.exp(-2 * x[negative]) + 1)) - 1
        return out

class GRU(nn.Module):
    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        self.input_size = fan_in
        self.hidden_size = fan_out

        # Gates: z, r, h̃
        self.x2z = Linear(fan_in, fan_out, bias)
        self.h2z = Linear(fan_out, fan_out, bias)

        self.x2r = Linear(fan_in, fan_out, bias)
        self.h2r = Linear(fan_out, fan_out, bias)

        self.x2h = Linear(fan_in, fan_out, bias)
        self.h2h = Linear(fan_out, fan_out, bias)

        self.sigmoid = Sigmoid()
        self.tanh = TanH()

    def forward(self, x, h0=None):
        """
        x: (seq_len, batch_size, input_size)
        h0: (batch_size, hidden_size)
        """
        seq_len, batch_size, _ = x.shape

        if h0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h0

        outputs = []

        for t in range(seq_len):
            x_t = x[t]

            z_t = self.sigmoid(self.x2z(x_t) + self.h2z(h_t))
            r_t = self.sigmoid(self.x2r(x_t) + self.h2r(h_t))
            h_candidate = self.tanh(self.x2h(x_t) + self.h2h(r_t * h_t))
            h_t = (1 - z_t) * h_t + z_t * h_candidate

            outputs.append(h_t.unsqueeze(0))  # (1, batch_size, hidden_size)

        out = torch.cat(outputs, dim=0)  # (seq_len, batch_size, hidden_size)
        return out, h_t.unsqueeze(0)

# --- Test Code ---
if __name__ == "__main__":
    # Test parameters
    seq_len = 5
    batch_size = 2
    input_size = 3
    hidden_size = 4

    # Random input (seq_len, batch_size, input_size)
    x = torch.randn(seq_len, batch_size, input_size)

    # Instantiate and run GRU
    gru = GRU(fan_in=input_size, fan_out=hidden_size)
    output, final_hidden = gru(x)

    # Print results
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Final hidden shape:", final_hidden.shape)

    print("\nSample output at t=0:\n", output[0])
    print("\nFinal hidden state:\n", final_hidden[0])