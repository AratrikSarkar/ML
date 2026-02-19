import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# DNA Vocabulary
DNA_VOCAB = ['A', 'C', 'G', 'T', 'N']
stoi = {ch: i for i, ch in enumerate(DNA_VOCAB)}
itos = {i: ch for ch, i in stoi.items()}
PAD_TOKEN = stoi['N']

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim, num_heads):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=2 * hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.fc_latent = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, x):
        x = self.embedding(x)                 # (B, L, E)
        lstm_out, _ = self.lstm(x)            # (B, L, 2H)

        pad_mask = (x.argmax(dim=-1) == PAD_TOKEN)  # (B, L)

        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=pad_mask
        )

        z = self.fc_latent(attn_out)  # (B, L, latent_dim)
        return z

class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_dim, hidden_dim, num_heads):
        super().__init__()

        self.lstm = nn.LSTM(
            latent_dim, hidden_dim,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, pad_mask=None):
        lstm_out, _ = self.lstm(z)
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=pad_mask
        )
        return self.fc_out(attn_out)

class DNAAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        pad_mask = (x == PAD_TOKEN)
        z = self.encoder(x)
        logits = self.decoder(z, pad_mask)
        return logits

def read_fasta(path):
    sequence = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            sequence.append(line.upper())
    dna = "".join(sequence)
    validate_dna(dna)
    return dna

def validate_dna(seq, allowed=None):
    if allowed is None:
        allowed = {'A', 'C', 'G', 'T', 'N'}
    invalid = set(seq) - allowed
    if invalid:
        raise ValueError(f"Invalid DNA symbols found: {invalid}")

def encode_dna(seq):
    return [stoi[ch] for ch in seq]

def decode_dna(tokens):
    return ''.join(itos[t] for t in tokens)

def chunk_sequence(tokens, block_size, pad_token=PAD_TOKEN):
    chunks = []
    for i in range(0, len(tokens), block_size):
        block = tokens[i:i + block_size]
        if len(block) < block_size:
            block = block + [pad_token] * (block_size - len(block))
        chunks.append(block)
    return chunks

def prepare_dataset(blocks):
    return torch.tensor(blocks, dtype=torch.long)

def reconstruction_loss(logits, targets):
    B, L, V = logits.shape
    logits = logits.view(B * L, V)
    targets = targets.view(B * L)

    return F.cross_entropy(
        logits,
        targets,
        ignore_index=PAD_TOKEN
    )

def train_autoencoder(model, dataset, epochs, batch_size, lr):
    model.to(device)
    dataset = dataset.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_samples = dataset.size(0)

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        total_loss = 0.0

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch = dataset[idx]

            logits = model(batch)
            loss = reconstruction_loss(logits, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (num_samples // batch_size)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Parameters
    block_size = 128
    emb_dim = 64
    hidden_dim = 128
    latent_dim = 64
    num_heads = 4
    batch_size = 32
    epochs = 6
    learning_rate = 3e-4

    # Pipeline
    dna_seq = read_fasta("AeCa.txt")
    tokens = encode_dna(dna_seq)
    blocks = chunk_sequence(tokens, block_size)
    dataset = prepare_dataset(blocks)

    encoder = Encoder(
        vocab_size=len(DNA_VOCAB),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_heads=num_heads
    )

    decoder = Decoder(
        vocab_size=len(DNA_VOCAB),
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )

    model = DNAAutoEncoder(encoder, decoder)

    # Training
    train_autoencoder(
        model=model,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate
    )

    # Testing
    model.eval()
    with torch.no_grad():
        sample = dataset[:1].to(device)
        logits = model(sample)
        recon = torch.argmax(logits, dim=-1).cpu().tolist()[0]

    original = decode_dna(sample.cpu().tolist()[0])
    reconstructed = decode_dna(recon)

    print("\nOriginal   : ", original[:60])
    print("Reconstructed: ", reconstructed[:60])
