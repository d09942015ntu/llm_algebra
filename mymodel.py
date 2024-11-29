import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleAttentionTransformer(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super(SingleAttentionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        # Learnable parameters for query, key, and value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Feedforward layer
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)

        # Normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def attention(self, Q, K, V):
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attention_weights, V)
        return output

    def forward(self, x):
        # Compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Apply attention
        attention_output = self.attention(Q, K, V)

        # Residual connection and normalization
        x = self.layer_norm1(attention_output)

        # Feedforward layer
        x = F.relu(self.fc1(x))
        ff_output = F.relu(self.fc2(x))

        # Second residual connection and normalization
        output = self.layer_norm2(ff_output)

        return output

# Toy example configuration
embed_dim = 8
seq_len = 5
batch_size = 2

# Create random input data
x = torch.randn(batch_size, seq_len, embed_dim)

# Instantiate the transformer and print output
model = SingleAttentionTransformer(embed_dim=embed_dim, seq_len=seq_len)
output = model(x)

print("Output:", output)
