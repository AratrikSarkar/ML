import torch
import torch.nn as nn
from Model1 import M1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
#------------------------------------------------------------
block_size=32
batch_size = 64 
n_embd = 64
hidden_size=4*n_embd
n_head = 8
eval_interval=20
max_iters=100
learning_rate = 3e-4
eval_iters = 50
#------------------------------------------------------------

with open("AeCa.txt",'r',encoding='UTF-8') as f:
    text=f.read()
f.close()
chars=sorted(list(set(text)))                                #Total number of different characters in the DNA sequence
vocab_size=len(chars)
#print(chars,vocab_size)

stoi={s:i for i,s in enumerate(chars)}
itos={i:s for i,s in enumerate(stoi)}

encode=lambda word:[stoi[s] for s in word]
decode=lambda num: ''.join([itos[i] for i in num])

data=torch.tensor(encode(text),dtype=torch.long)
print(data.shape,data.dtype)

l=int(0.8*data.shape[0])
h=int(0.9*data.shape[0])
train_data=data[:l]
val_data=data[l:h]
test_data=data[h:]
#print(train_data.shape,val_data.shape,test_data.shape)


def get_batch(split):
    #generate a small batch of data of inputs x and target y
    data =train_data if split=='train' else val_data
    ix= torch.randint(len(data)-block_size,(batch_size,))  # if len(data) is 10 we cannot find a chunck of size 9 after index 1
    #print(ix)
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y

x,y=get_batch('train')
x = x.to(device)
y = y.to(device)
print(x.shape,y.shape)
model=M1(block_size,vocab_size,n_embd,hidden_size,n_head)
model=model.to(device)
z=model(x)
print(z[0].shape)


def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out


optimizer=torch.optim.Adam(model.parameters(),learning_rate)
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    #print(xb.shape,yb.shape)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

