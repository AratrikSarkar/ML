import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Input → Gates
        self.x2i = Linear(input_size, hidden_size, bias)  # input gate
        self.x2f = Linear(input_size, hidden_size, bias)  # forget gate
        self.x2o = Linear(input_size, hidden_size, bias)  # output gate
        self.x2g = Linear(input_size, hidden_size, bias)  # candidate cell state

        # Hidden → Gates
        self.h2i = Linear(hidden_size, hidden_size, bias)
        self.h2f = Linear(hidden_size, hidden_size, bias)
        self.h2o = Linear(hidden_size, hidden_size, bias)
        self.h2g = Linear(hidden_size, hidden_size, bias)

        # Activations
        self.sigmoid = Sigmoid()
        self.tanh = TanH()

    def forward(self, x, state=None):
        """
        x: (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
        state: tuple (h0, c0)
               h0: (batch_size, hidden_size)
               c0: (batch_size, hidden_size)
        """
        if self.batch_first:
            x = x.permute(1, 0, 2)  # to (seq_len, batch_size, input_size)

        seq_len, batch_size, _ = x.shape

        # h_t : Hidden State (Short Term Memory)
        # c_t : Cell State (Long Term Memory)

        if state is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = state

        outputs = torch.zeros(seq_len, batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t = x[t]

            # Gates
            i_t = self.sigmoid(self.x2i(x_t) + self.h2i(h_t))   # input gate
            f_t = self.sigmoid(self.x2f(x_t) + self.h2f(h_t))   # forget gate
            o_t = self.sigmoid(self.x2o(x_t) + self.h2o(h_t))   # output gate
            g_t = self.tanh(self.x2g(x_t) + self.h2g(h_t))      # candidate cell state

            # Cell state update
            c_t = f_t * c_t + i_t * g_t

            # Hidden state update
            h_t = o_t * self.tanh(c_t)

            outputs[t] = h_t

        if self.batch_first:
            outputs = outputs.permute(1, 0, 2)

        return outputs, (h_t.unsqueeze(0), c_t.unsqueeze(0))

if __name__ == '__main__':
    input_size = 5
    hidden_size = 3
    seq_len = 4
    batch_size = 2

    lstm = LSTM(input_size, hidden_size, batch_first=True)
    x = torch.randn(batch_size, seq_len, input_size)

    outputs, (h_n, c_n) = lstm(x)
    print("Outputs:", outputs.shape)  # (batch, seq_len, hidden_size)
    print("h_n:", h_n.shape)  # (1, batch, hidden_size)
    print("c_n:", c_n.shape)  # (1, batch, hidden_size)