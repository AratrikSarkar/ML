import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Linear(nn.Module):
    def __init__(self,fan_in,fan_out,bias=False):
        super().__init__()
        self.W=nn.Parameter(torch.randn((fan_out,fan_in)))
        self.b=nn.Parameter(torch.randn(fan_out)) if bias==True else None

    def forward(self, x):
        out= x @ self.W.T
        if self.b is not None:
            out=out+self.b
        return  out

class ReLu(nn.Module):
    def __init__(self,inplace=False):
        super().__init__()
        self.inplace=inplace

    def forward(self,x):
        if(self.inplace):
            return torch.clamp_(x,min=0.0)      # modifies x
        else:
            return torch.clamp(x,min=0.0)       # x remains unchanged

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out=torch.empty_like(x)
        positive=x>=0
        negative=~positive

        # For x ≥ 0:
        out[positive]=1/(1+torch.exp(-x[positive]))

        # For x < 0: reformulated to avoid large exp
        exp_x=torch.exp(x[negative])
        out[negative]=exp_x/(1+exp_x)
        return out

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

class TanH(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        out=torch.empty_like(x)
        positive=x>=0
        negative=~positive
        out[positive]=1-(2/(torch.exp(2*x[positive])+1))
        out[negative]=(2/(torch.exp(-2*x[negative])+1))-1
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
            return x.mul_(mask).div_(1 - self.p)  # scale to keep expected value
        else:
            return (x * mask) / (1 - self.p)

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

        new_shape = x.shape[:start] + (-1,) + (() if end == nd - 1 else x.shape[end+1:])

        return x.reshape(new_shape)

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,bias=True,batch_first=False):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.batch_first=batch_first

        self.i2h=Linear(input_size,hidden_size,bias)         #input ----> hidden
        self.h2h=Linear(hidden_size,hidden_size,bias)        #hidden ---> hidden
        self.activation=TanH()

    def forward(self, x, h0=None):
        """
        x: (seq_len, batch_size, input_size)
        h0: (batch_size, hidden_size)
        """
        if self.batch_first:
            x=x.permute(1,0,2)    # (batch_size, seq_len, input_size) ----> (seq_len, batch_size, input_size)


        seq_len, batch_size, _= x.shape

        if h0 is None:
            h_t=torch.zeros(batch_size, self.hidden_size,device=x.device)
        else:
            h_t=h0

        outputs=torch.zeros(seq_len,batch_size, self.hidden_size,device=x.device)
        for t in range(seq_len):
            x_t=x[t]            #(batch_size,input_size)
            # (batch_size,input_size)  @ (input_size,hidden_size) ----> (batch_size,hidden_size)
            # (batch_size,hidden_size)  @ (hidden_size,hidden_size) ----> (batch_size,hidden_size)
            h_t=self.activation(self.i2h(x_t)+self.h2h(h_t)) # (batch_size,hidden_size) + (batch_size,hidden_size)
            outputs[t]=h_t

        if self.batch_first:
            outputs=outputs.permute(1,0,2) # (seq_len, batch_size, input_size) ----> (batch_size, seq_len, input_size)

        return  outputs,h_t.unsqueeze(0)    #for h_t: (batch_size,hidden_size) ----> (1,batch_size,hidden_size)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True,batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first=batch_first

        # Gates: z, r, h̃
        # Update Gate Layers
        self.x2z = Linear(input_size, hidden_size, bias)
        self.h2z = Linear(hidden_size, hidden_size, bias)

        # Reset Gate Layers
        self.x2r = Linear(input_size, hidden_size, bias)
        self.h2r = Linear(hidden_size, hidden_size, bias)

        # Candidate Hidden State Layers
        self.x2h = Linear(input_size, hidden_size, bias)
        self.h2h = Linear(hidden_size, hidden_size, bias)

        self.sigmoid = Sigmoid()
        self.tanh = TanH()

    def forward(self, x, h0=None):
        """
        x: (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
        h0: (batch_size, hidden_size)
        """
        if self.batch_first:
            x=x.permute(1,0,2)                  # (batch_size, seq_len, input_size) ----> (seq_len, batch_size, input_size)

        seq_len, batch_size, _ = x.shape

        # z_t : Update Gate
        # r_t : Reset Gate
        # h_t : Hidden Gate

        if h0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h0

        outputs = torch.zeros(seq_len,batch_size,self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t = x[t]

            z_t = self.sigmoid(self.x2z(x_t) + self.h2z(h_t))
            r_t = self.sigmoid(self.x2r(x_t) + self.h2r(h_t))
            h_candidate = self.tanh(self.x2h(x_t) + self.h2h(r_t * h_t))    # candidate hidden state
            h_t = (1 - z_t) * h_t + z_t * h_candidate

            outputs[t]=h_t

        if self.batch_first:
            outputs=outputs.permute(1,0,2) # (seq_len, batch_size, input_size) ----> (batch_size, seq_len, input_size)

        return outputs, h_t.unsqueeze(0)    #for h_t: (batch_size,hidden_size) ----> (1,batch_size,hidden_size)

class Head(nn.Module):
    """ a single self attention head """

    def __init__(self, n_embd, head_size, block_size,bias=False):
        super().__init__()
        self.key=Linear(n_embd,head_size,bias)
        self.query=Linear(n_embd,head_size,bias)
        self.value=Linear(n_embd,head_size,bias)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=Dropout().to(device=device)
        self.softmax=Softmax(dim=-1)

    def forward(self,x):

        B,T,C = x.shape
        k = self.key(x)       # (B,T,hs)
        q = self.query(x)     # (B,T,hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1)*C**(-0.5)                               #(B,T,C) @ (B,C,T) --------> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))         #(B,T,T)
        wei = self.softmax(wei)                                        #(B,T,T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out,wei

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

class BatchNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (scale & shift)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running statistics (for eval mode)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # mean & var across batch + H + W
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            # update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # use running stats during eval
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        # normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
        return out

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        # Learnable params
        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
        self.beta = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        # Normalize across last len(normalized_shape) dims
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        return out

if __name__ == "__main__":
    # ==== Test Linear ====
    print("Linear Test:")
    q=Linear(3,4)
    m=q.to(device)
    x=torch.randn((4,3)).to(device)
    print(x.shape)
    print(m(x))

    # ==== Test ReLU ====
    print("ReLU Test:")
    a=torch.randn((1,4)).to(device)
    print(a,torch.relu(a))
    k=ReLu().to(device)
    print(k(a),a)

    # ==== Test Sigmoid ====
    print("Sigmoid Test:")
    x = torch.tensor([-1000.0, 0.0, 1000.0]).to(device=device)
    k=Sigmoid().to(device=device)
    print("Stable Sigmoid:", k(x))

    # ==== Test TanH ====
    print("TanH Test:")
    x = torch.tensor([-1000.0, 0.0, 1000.0]).to(device=device)
    k=TanH().to(device=device)
    print("Stable TanH:", k(x))

    # ==== Test RNN ====
    print("RNN Test:")
    seq_len, batch_size, input_size, hidden_size = 5, 3, 10, 8
    x = torch.randn(seq_len, batch_size, input_size).to(device=device)
    rnn = RNN(input_size, hidden_size).to(device=device)

    out, hn = rnn(x)
    print(x.shape)
    print(out.shape)  # torch.Size([5, 3, 8])
    print(hn.shape)   # torch.Size([1, 3, 8])

    # ==== Test GRU ====
    print("GRU Test:")
    seq_len = 5
    batch_size = 2
    input_size = 3
    hidden_size = 4

    # Random input (seq_len, batch_size, input_size)
    x = torch.randn(seq_len, batch_size, input_size)

    # Instantiate and run GRU
    gru = GRU(input_size=input_size, hidden_size=hidden_size)
    output, final_hidden = gru(x)

    # Print results
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Final hidden shape:", final_hidden.shape)

    print("\nSample output at t=0:\n", output[0])
    print("\nFinal hidden state:\n", final_hidden[0])

    # ==== Test Softmax ====
    print("Softmax Test:")
    softmax = Softmax(dim=1)
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [0.5, 0.2, 0.3]])
    print("Input to Softmax:\n", x)
    print("Softmax output:\n", softmax(x))
    print("Sum along dim=1 (should be 1):\n", softmax(x).sum(dim=1))

    # ==== Test Dropout ====
    print("Dropout Test:")
    dropout = Dropout(p=0.3)
    dropout.train()  # enable dropout mode

    y = torch.ones((5, 5))
    print("\nInput to Dropout:\n", y)
    print("Dropout output (train mode):\n", dropout(y))

    dropout.eval()  # disable dropout
    print("\nDropout output (eval mode - should be unchanged):\n", dropout(y))

    # ==== Test Flatten ====
    print("Flatten Test:")
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

    # ==== Test Attention Head ====
    # Define the hyperparameters for the self-attention head
    n_embd = 64        # Embedding dimension (input size for the linear layers)
    head_size = 16     # Head size (output size for the linear layers)
    block_size = 8     # Sequence length
    batch_size = 4     # Number of sequences in the batch
    print("\nAttention Head test:")
    head = Head(n_embd, head_size,block_size,False).to(device=device)

    # Create a random input tensor
    x = torch.randn(batch_size, block_size, n_embd).to(device=device)

    # Forward pass
    out, wei = head(x)

    # Print the shapes
    print(f"Input shape (x): {x.shape}")
    print(f"Output shape (out): {out.shape}")
    print(f"Attention weights shape (wei): {wei.shape}")

    # ==== Test LSTM ====
    print("LSTM Test:")
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

    # ==== Test CNN ====
    print("CNN Test:")

    B, C_in, H, W = 2, 3, 5, 5  # batch=2, channels=3, height=5, width=5
    x = torch.randn(B, C_in, H, W).to(device)

    conv = Conv2D(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1).to(device)
    pool = MaxPool2D(kernel_size=2, stride=2).to(device)

    y = conv(x)
    print("Conv2D output shape:", y.shape)  # Expect: (B, out_channels, H, W)

    z = pool(y)
    print("MaxPool2D output shape:", z.shape)  # Expect: (B, out_channels, H/2, W/2)

    # ==== Test BatchNorm ====
    bn = BatchNorm2D(3)
    x = torch.randn(8, 3, 32, 32)  # B, C, H, W
    y = bn(x)
    print("BatchNorm2D out:", y.shape)

    # ==== Test LayerNorm ====
    ln = LayerNorm(10)  # normalize last dim
    x = torch.randn(4, 5, 10)
    y = ln(x)
    print("LayerNorm out:", y.shape)