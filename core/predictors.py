import torch
import torch.nn as nn
from core.attentions import MultiHeadedAttention

class PitchPredictor(nn.Module):
    def __init__(self, n_layers=30, kernel_size=3, n_att_layers=10, n_heads=8, hidden_size=512, dropout=0.5,
                 attn_dropout=0.2, conv_in_stacks=3):
        super(PitchPredictor, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
            for _ in range(n_layers)
        ])

        self.attention_layers = nn.ModuleList([
            MultiHeadedAttention(n_head = n_heads, n_feat = hidden_size, dropout_rate = attn_dropout)
            for _ in range(n_att_layers)
        ])

        self.fc = nn.Linear(hidden_size, 1)
        self.conv_in_stacks = conv_in_stacks

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prompt_encoder_output):
        # x shape: (batch_size, hidden_size, length)
        # prompt_encoder_output shape: (batch_size, prompt_length, hidden_size)

        # Transpose prompt_encoder_output to (batch_size, hidden_size, prompt_length)

        # Apply conv layers
        n = 0
        x = x.transpose(1, 2)
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = torch.relu(x)
            x = self.dropout(x)
            if (i+1) % self.conv_in_stacks == 0:
                # Apply Q-K-V attention layer

                x = x + self.attention_layers[n](x.transpose(1, 2), prompt_encoder_output,
                                                 prompt_encoder_output).transpose(1, 2)
                n = n + 1

        # Apply fully connected layer
        x = self.fc(x.transpose(1, 2)).squeeze(-1)

        # Make sure predicted pitches are positive
        x = torch.relu(x)

        return x


class DurationPredictor(nn.Module):
    def __init__(self, n_layers=30, kernel_size=3, n_att_layers=10, n_heads=8, hidden_size=512, dropout=0.2,
                 attn_dropout=0.2, conv_in_stacks=3):
        super(DurationPredictor, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
            for _ in range(n_layers)
        ])

        self.attention_layers = nn.ModuleList([
            MultiHeadedAttention(n_head=n_heads, n_feat=hidden_size, dropout_rate=attn_dropout)
            for _ in range(n_att_layers)
        ])

        self.fc = nn.Linear(hidden_size, 1)
        self.conv_in_stacks = conv_in_stacks

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prompt_encoder_output):
        # x shape: (batch_size, hidden_size, length)
        # prompt_encoder_output shape: (batch_size, prompt_length, hidden_size)

        # Apply conv layers
        n = 0
        x = x.transpose(1, 2)
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = torch.relu(x)
            x = self.dropout(x)
            if (i + 1) % self.conv_in_stacks == 0:
                # Apply Q-K-V attention layer

                x = x + self.attention_layers[n](x.transpose(1, 2), prompt_encoder_output,
                                                 prompt_encoder_output).transpose(1, 2)
                n = n + 1

        # Apply fully connected layer
        x = self.fc(x.transpose(1, 2)).squeeze(-1)

        # Make sure predicted pitches are positive
        x = torch.relu(x)

        return x