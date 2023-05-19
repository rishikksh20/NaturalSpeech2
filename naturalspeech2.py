import torch.nn as nn
import torch
import math
from core.embedding import PositionalEncoding, ScaledPositionalEncoding
from core.phoneme_encoder import Encoder
from core.diffusion import Diffusion
from core.predictors import DurationPredictor, PitchPredictor
from utils.util import make_non_pad_mask, make_pad_mask
from core.length_regulator import LengthRegulator
from core.decoder import DiffusionDecoder

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class NaturalSpeech2(nn.Module):

    def __init__(self, idim, config, codec, padding_idx=0):

        super().__init__()

        # get positional encoding class
        pos_enc_class = ScaledPositionalEncoding


        # define encoder
        # encoder_input_layer = torch.nn.Embedding(
        #     num_embeddings=idim, embedding_dim=hp.model.adim, padding_idx=padding_idx
        # )
        self.token_embedding = nn.Embedding(
            num_embeddings=idim,
            embedding_dim=config.model.adim,
        )
        embedding_variance = math.sqrt(3.0) * math.sqrt(2.0 / (idim + config.model.adim))
        self.token_embedding.weight.data.uniform_(-embedding_variance, embedding_variance)

        self.phoneme_encoder = Encoder(
            idim=config.speech_prompt.adim,
            num_blocks=config.model.elayers,
            attention_heads=config.model.aheads,
            attention_dim=config.model.adim,
            linear_units=config.model.eunits,
            positionwise_conv_kernel_size=config.model.positionwise_conv_kernel_size,
            pos_enc_class=pos_enc_class,
            input_layer=None,
            attention_dropout_rate=config.model.encoder_dropout
        )

        self.duration_predictor = DurationPredictor(
            n_layers=config.model.duration_predictor_layers,
            kernel_size=config.model.duration_predictor_kernel_size,
            n_att_layers=config.model.duration_attn_layers,
            n_heads=config.model.duration_aheads,
            hidden_size=config.model.duration_predictor_chans,
            dropout=config.model.duration_predictor_dropout_rate
        )
        self.length_regulator = LengthRegulator()

        self.pitch_predictor = PitchPredictor(
            n_layers=config.model.pitch_predictor_layers,
            kernel_size=config.model.pitch_predictor_kernel_size,
            n_att_layers=config.model.pitch_attn_layers,
            n_heads=config.model.pitch_aheads,
            hidden_size=config.model.pitch_predictor_chans,
            dropout=config.model.pitch_predictor_dropout_rate
        )
        self.pitch_embed = torch.nn.Linear(1, config.model.adim)

        # self.speech_prompt = SpeechPromptEncoder(
        #     idim=config.speech_prompt.idim,
        #     adim=config.speech_prompt.adim,
        #     kernel_size=config.speech_prompt.kernel_size,
        #     padding=config.speech_prompt.padding,
        #     nheads=config.speech_prompt.nheads,
        #     num_layers=config.speech_prompt.num_layers,
        #     ff_units=config.speech_prompt.ff_units,
        #     dropout=config.speech_prompt.dropout
        # )

        self.speech_prompt = Encoder(
            idim=config.speech_prompt.idim,
            num_blocks=config.model.elayers,
            attention_heads=config.model.aheads,
            attention_dim=config.model.adim,
            linear_units=config.model.eunits,
            positionwise_conv_kernel_size=config.model.positionwise_conv_kernel_size,
            pos_enc_class=pos_enc_class,
            attention_dropout_rate=config.model.encoder_dropout
        )


        decoder = DiffusionDecoder(config.model.adim, config)
        self.diffusion = Diffusion(model=decoder, codec=codec, config=config)


    def forward(self, tokens, token_lengths, speech_prompts, audio, codes, output_length,
                ds, ps, inference=False):
        '''

        :param tokens:  [B, Lmax, Dim]
        :param token_lengths: [B]
        :param speech_prompts: [B, T, hidden]
        :param output_length: [B]
        :param dur: [B, Lmax]
        :param pitch: [B, Tmax]
        :param audio: [B, Tmax, Dim]
        :param codes: [B, Tmax, num_quantizers]
        :return:
        '''

        # forward encoder
        x_masks = self._source_mask(
            token_lengths
        )  # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])
        tokens = self.token_embedding(tokens)
        hs, _ = self.phoneme_encoder(
            tokens, x_masks
        )  # (B, Tmax, adim) -> torch.Size([32, 121, 256])
        # print("ys :", ys.shape)

        prompt_mask = None

        prompt, _ = self.speech_prompt(
            speech_prompts, prompt_mask
        )

        d_masks = make_pad_mask(token_lengths).to(tokens.device)

        d_outs = self.duration_predictor(hs, prompt)

        if inference:
            hs = self.length_regulator(hs, d_outs, token_lengths)  # (B, Lmax, adim)
            p_outs = self.pitch_predictor(hs.detach(), prompt)  # torch.Size([32, 868])
            hs = hs + self.pitch_embed(p_outs.unsqueeze(-1))
        else:
            hs = self.length_regulator(hs, ds, token_lengths)  # (B, Lmax, adim)

            p_outs = self.pitch_predictor(hs.detach(), prompt)  # torch.Size([32, 868])

            hs = hs + self.pitch_embed(ps.unsqueeze(-1))

        out, diff_loss, rvq_ce_loss = self.diffusion(hs.transpose(1, -1), prompt, output_length,
                                                     audio, codes)




        return out, p_outs, d_outs, diff_loss, rvq_ce_loss

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.
        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(device=next(self.parameters()).device)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)






if __name__ == '__main__':
    from utils.hparams import HParam
    from codec.encodec import EncodecWrapper
    idim = 126

    config_path = "configs/config.yaml"  # you can change it to anything else
    config = HParam(config_path)

    codec = EncodecWrapper()

    model = NaturalSpeech2(
        idim,
        config,
        codec
    )
    tokens = torch.ones([2, 100]).to(torch.long)
    token_len = torch.tensor([100, 100])
    speech_prompts = torch.ones([2, 10, 128])
    audio = torch.ones([2, 100, 128])
    codes = torch.ones([2, 100, 8]).to(torch.long)
    olens = torch.tensor([100, 100])
    ds = torch.ones([2, 100])
    ps = torch.ones([2, 100])
    model(tokens, token_len, speech_prompts, audio, codes, olens, ds, ps)


