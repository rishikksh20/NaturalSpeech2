import math
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.attentions import MultiHeadedAttention
from einops import rearrange

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

def exists(x):
    return x is not None
def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)



class FiLM(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()

    self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
    self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.input_conv.weight)
    nn.init.xavier_uniform_(self.output_conv.weight)
    nn.init.zeros_(self.input_conv.bias)
    nn.init.zeros_(self.output_conv.bias)

  def forward(self, x):
    x = self.input_conv(x)
    x = F.leaky_relu(x, 0.2)
    shift, scale = torch.chunk(self.output_conv(x), 2, dim=1)
    return shift, scale



class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation, apply_film, n_heads, attn_dropout):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

        self.apply_film = False
        if apply_film:
            self.attention = MultiHeadedAttention(
                n_head=n_heads, n_feat=encoder_hidden, dropout_rate=attn_dropout
            )
            self.film = FiLM(
                output_size=2 * residual_channels,
                input_size=encoder_hidden,
            )
            self.apply_film = True

    def forward(self, x, conditioner, diffusion_step, speech_prompt):
        # x : [B, hidden, T]
        # speech_prompt : [B, T, hidden]
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step

        if self.apply_film:
            attn = self.attention(y.transpose(1, 2), speech_prompt, speech_prompt).transpose(1, 2)
            shift, scale = self.film(attn)

        y = self.dilated_conv(y) + conditioner

        if self.apply_film:
            y = scale * y + shift

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffusionDecoder(nn.Module):
    def __init__(self, in_dims, hp):
        super().__init__()

        encoder_hidden = in_dims
        self.out_dims = hp.codec.dim

        residual_channels = hp.decoder.adim
        dilation_cycle_length = hp.model.dilation_cycle_length


        self.qkv_attention_query = torch.nn.Parameter(
            torch.empty(1, hp.model.qkv_num_tokens, hp.speech_prompt.adim)
        )
        query_variance = math.sqrt(3.0) * math.sqrt(
            2.0 / (hp.speech_prompt.adim + hp.model.qkv_num_tokens))
        self.qkv_attention_query.data.uniform_(-query_variance, query_variance)

        self.qkv_attn = MultiHeadedAttention(
            n_head=hp.model.qkv_heads, n_feat=hp.model.adim, dropout_rate=hp.decoder.attn_dropout
        )

        self.input_projection = Conv1d(in_dims, residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(residual_channels)
        dim = residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(encoder_hidden, residual_channels, 2 ** (i % dilation_cycle_length),
                          apply_film=(i + 1) % hp.decoder.wavenet_stack == 0,
                          n_heads=hp.decoder.nheads, attn_dropout=hp.decoder.attn_dropout)
            for i in range(hp.decoder.num_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, self.out_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond, speech_prompt):
        """
        :param spec: [B, Tmax, Dim]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :param speech_prompt : [B, T, hidden]
        :return:
        """
        x = spec.transpose(1, -1)
        x = self.input_projection(x)  # x [B, residual_channel, T]
        # qkv_attention_query [1, 512, 32 ]
        speech_prompt = self.qkv_attn(self.qkv_attention_query.expand(speech_prompt.size(0), -1, -1),
                                     speech_prompt, speech_prompt)
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step, speech_prompt)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, codebook_dim, T]
        return x.transpose(-2, -1)