import torch
import math
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from functools import partial

from torch import expm1, nn
from tqdm import tqdm

from core.decoder import DiffusionDecoder


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_div(numer, denom):
    return numer / denom.clamp(min = 1e-10)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr

def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)

# gaussian diffusion

class Diffusion(nn.Module):

    def __init__(
        self,
        model: DiffusionDecoder,
        codec = None,
        config = None,
                                     # this will be set to < 1. for better convergence when training on higher resolution images
    ):
        super().__init__()
        self.model = model
        self.codec = codec

        target_sample_hz = config.diff.target_sample_hz
        timesteps = config.diff.timesteps
        use_ddim = config.diff.use_ddim
        noise_schedule = config.diff.noise_schedule
        objective = config.diff.objective
        schedule_kwargs = config.diff.schedule_config
        time_difference = config.diff.time_difference
        min_snr_loss_weight = config.diff.min_snr_loss_weight
        min_snr_gamma = config.diff.min_snr_gamma
        train_prob_self_cond = config.diff.train_prob_self_cond
        rvq_cross_entropy_loss_weight = config.diff.rvq_cross_entropy_loss_weight  # default this to off until we are sure it is working. not totally sold that this is critical
        scale = config.diff.scale

        assert exists(codec) or exists(target_sample_hz)

        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = None

        if exists(codec):
            self.target_sample_hz = codec.target_sample_hz
            self.seq_len_multiple_of = codec.seq_len_multiple_of

        assert not exists(codec) or model.out_dims == codec.codebook_dim, f'transformer model dimension {model.dim} must be equal to codec dimension {codec.codebook_dim}'

        self.dim = codec.codebook_dim if exists(codec) else model.out_dims

        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective

        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # the main finding presented in Ting Chen's paper - that higher resolution images requires more noise for better training

        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale

        # gamma schedules

        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

        # probability for self conditioning during training

        self.train_prob_self_cond = train_prob_self_cond

        # min snr loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        # weight of the cross entropy loss to residual vq codebooks

        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight

    @property
    def device(self):
        return next(self.model.parameters()).device

    def print(self, s):
        return self.accelerator.print(s)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, cond, prompt, shape, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        audio = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min = 0.)

            noise_cond = time

            # get predicted x0

            model_output = self.model(audio, noise_cond, cond, prompt)

            # get log(snr)

            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            # get alpha sigma of time and next time

            alpha, sigma = gamma_to_alpha_sigma(gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next, self.scale)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output

            # derive posterior mean and variance

            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (audio * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(audio),
                torch.zeros_like(audio)
            )

            audio = mean + (0.5 * log_variance).exp() * noise

        return audio

    @torch.no_grad()
    def ddim_sample(self, cond, prompt, shape, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        audio = torch.randn(shape, device = device)

        x_start = None
        last_latents = None

        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # get times and noise levels

            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min = 0.)

            # predict x0

            model_output = self.model(audio, times, cond, prompt)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output

            # get predicted noise

            pred_noise = safe_div(audio - alpha * x_start, sigma)

            # calculate x next

            audio = x_start * alpha_next + pred_noise * sigma_next

        return audio

    @torch.no_grad()
    def sample(
        self,
        cond,
        prompt,
        length,
        batch_size = 1
    ):
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        audio = sample_fn(cond, prompt, (batch_size, length, self.dim))

        if exists(self.codec):
            audio = self.codec.decode(audio)

            if audio.ndim == 3:
                audio = rearrange(audio, 'b 1 n -> b n')

        return audio

    def forward(
        self,
        cond,
        prompt,
        olens,
        audio,
        codes = None,
    ):

        batch, n, d, device = *audio.shape, self.device

        assert d == self.dim, f'codec codebook dimension {d} must match model dimensions {self.dim}'

        # sample random times

        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)

        # noise sample

        noise = torch.randn_like(audio)

        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(audio, gamma)
        alpha, sigma =  gamma_to_alpha_sigma(padded_gamma, self.scale)

        noised_audio = alpha * audio + sigma * noise

        # predict and take gradient step

        pred = self.model(noised_audio, times, cond, prompt)

        if self.objective == 'eps':
            target = noise

        elif self.objective == 'x0':
            target = audio

        elif self.objective == 'v':
            target = alpha * noise - sigma * audio

        loss = F.mse_loss(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # min snr loss weight

        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = self.min_snr_gamma)

        if self.objective == 'eps':
            loss_weight = maybe_clipped_snr / snr

        elif self.objective == 'x0':
            loss_weight = maybe_clipped_snr

        elif self.objective == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        loss =  (loss * loss_weight).mean()

        # cross entropy loss to codebooks

        if self.objective == 'x0':
            x_start = pred

        elif self.objective == 'eps':
            x_start = safe_div(audio - sigma * pred, alpha)

        elif self.objective == 'v':
            x_start = alpha * audio - sigma * pred


        if self.rvq_cross_entropy_loss_weight == 0 or not exists(codes):
            return x_start, loss, 0.0

        _, ce_loss = self.codec.rq(x_start, codes)

        return x_start, loss , self.rvq_cross_entropy_loss_weight * ce_loss