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
    
class RNN(nn.Module):
    def __init__(self,fan_in,fan_out,bias=True,batch_first=False):
        super().__init__()
        self.input_size=fan_in
        self.hidden_size=fan_out
        self.batch_first=batch_first
        
        self.i2h=Linear(fan_in,fan_out,bias)         #input ----> hidden
        self.h2h=Linear(fan_out,fan_out,bias)        #hidden ---> hidden
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
    def __init__(self, fan_in, fan_out, bias=True,batch_first=False):
        super().__init__()
        self.input_size = fan_in
        self.hidden_size = fan_out
        self.batch_first=batch_first

        # Gates: z, r, h̃
        # Update Gate Layers
        self.x2z = Linear(fan_in, fan_out, bias)
        self.h2z = Linear(fan_out, fan_out, bias)

        # Reset Gate Layers
        self.x2r = Linear(fan_in, fan_out, bias)
        self.h2r = Linear(fan_out, fan_out, bias)

        # Candidate Hidden State Layers
        self.x2h = Linear(fan_in, fan_out, bias)
        self.h2h = Linear(fan_out, fan_out, bias)

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

if __name__ == "__main__":
    q=Linear(3,4)
    m=q.to(device)
    x=torch.randn((4,3)).to(device)
    print(x.shape)
    print(m(x))

    a=torch.randn((1,4)).to(device)
    print(a,torch.relu(a))
    k=ReLu().to(device)
    print(k(a),a)

    x = torch.tensor([-1000.0, 0.0, 1000.0]).to(device=device)
    k=Sigmoid().to(device=device)
    print("Stable Sigmoid:", k(x))

    x = torch.tensor([-1000.0, 0.0, 1000.0]).to(device=device)
    k=TanH().to(device=device)
    print("Stable TanH:", k(x))
    
    seq_len, batch_size, input_size, hidden_size = 5, 3, 10, 8
    x = torch.randn(seq_len, batch_size, input_size).to(device=device)
    rnn = RNN(input_size, hidden_size).to(device=device)
    
    out, hn = rnn(x)
    print(x.shape)
    print(out.shape)  # torch.Size([5, 3, 8])
    print(hn.shape)   # torch.Size([1, 3, 8])   

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