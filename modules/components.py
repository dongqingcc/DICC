import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from inspect import isfunction
from torch.autograd import Function

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


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


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.conv = nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.conv = nn.Conv2d(dim_in, dim_out, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, large=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 7 if large else 3, padding=3 if large else 1), LayerNorm(dim_out), nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, large=False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, large)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(time_emb):
            h = h + self.mlp(time_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None):
        super().__init__()
        if dim_head is None:
            dim_head = dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, n_layer=1):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.cur_states = [None for i in range(n_layer)]
        self.n_layer = n_layer

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.input_dim + self.hidden_dim,
                    out_channels=4 * self.hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=self.bias,
                )
            ]
            + [
                nn.Conv2d(
                    in_channels=self.hidden_dim + self.hidden_dim,
                    out_channels=4 * self.hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=self.bias,
                )
                for i in range(n_layer - 1)
            ]
        )

    def step_forward(self, input_tensor, layer_index=0):
        assert self.cur_states[layer_index] is not None
        h_cur, c_cur = self.cur_states[layer_index]
        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.convs[layer_index](combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        self.cur_states[layer_index] = (h_next, c_next)

        return h_next

    def forward(self, input_tensor):
        for i in range(self.n_layer):
            input_tensor = self.step_forward(input_tensor, i)
        return input_tensor

    def init_hidden(self, batch_shape):
        B, _, H, W = batch_shape
        for i in range(self.n_layer):
            self.cur_states[i] = (
                torch.zeros(B, self.hidden_dim, H, W, device=self.convs[0].weight.device,),
                torch.zeros(B, self.hidden_dim, H, W, device=self.convs[0].weight.device,),
            )


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, n_layer=1):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super().__init__()
        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.cur_states = [None for _ in range(n_layer)]
        self.n_layer = n_layer
        self.conv_gates = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

        self.conv_cans = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=self.hidden_dim,  # for candidate neural memory
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

    def init_hidden(self, batch_shape):
        b, _, h, w = batch_shape
        for i in range(self.n_layer):
            self.cur_states[i] = torch.zeros((b, self.hidden_dim, h, w), device=self.conv_cans[0].weight.device)

    def step_forward(self, input_tensor, index):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        h_cur = self.cur_states[index]
        assert h_cur is not None
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates[index](combined)

        reset_gate, update_gate = torch.split(torch.sigmoid(combined_conv), self.hidden_dim, dim=1)
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_cans[index](combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        self.cur_states[index] = h_next
        return h_next
    
    def forward(self, input_tensor):
        for i in range(self.n_layer):
            input_tensor = self.step_forward(input_tensor, i)
        return input_tensor


class VBRCondition(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.scale = nn.Conv2d(input_dim, output_dim, 1)
        self.shift = nn.Conv2d(input_dim, output_dim, 1)

    def forward(self, input, cond):
        cond = cond.reshape(-1, 1, 1, 1)
        scale = self.scale(cond)
        shift = self.shift(cond)
        return input * scale + shift


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
    def __init__(self, ch, inverse=False, beta_min=1e-6, gamma_init=.1, reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class GDN1(GDN):
    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(torch.abs(inputs), gamma, beta)
        # norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class PriorFunction(nn.Module):
    #  A Custom Function described in Balle et al 2018. https://arxiv.org/pdf/1802.01436.pdf
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, parallel_dims, in_features, out_features, scale, bias=True):
        super(PriorFunction, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.5, 0.5)

    def forward(self, input, detach=False):
        # input shape (channel, batch_size, in_features)
        if detach:
            return torch.matmul(input, F.softplus(self.weight.detach())) + self.bias.detach()
        return torch.matmul(input, F.softplus(self.weight)) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias
                                                                 is not None)


class FlexiblePrior(nn.Module):
    '''
        A prior model described in Balle et al 2018 Appendix 6.1 https://arxiv.org/pdf/1802.01436.pdf
        return the boxshape likelihood
    '''
    def __init__(self, channels=256, dims=[3, 3, 3], init_scale=10.):
        super(FlexiblePrior, self).__init__()
        dims = [1] + dims + [1]
        self.chain_len = len(dims) - 1
        scale = init_scale**(1 / self.chain_len)
        h_b = []
        for i in range(self.chain_len):
            init = np.log(np.expm1(1 / scale / dims[i + 1]))
            h_b.append(PriorFunction(channels, dims[i], dims[i + 1], init))
        self.affine = nn.ModuleList(h_b)
        self.a = nn.ParameterList(
            [nn.Parameter(torch.zeros(channels, 1, 1, 1, dims[i + 1])) for i in range(self.chain_len - 1)])

        # optimize the medians to fix the offset issue
        self._medians = nn.Parameter(torch.zeros(1, channels, 1, 1))
        # self.register_buffer('_medians', torch.zeros(1, channels, 1, 1))

    @property
    def medians(self):
        return self._medians.detach()

    def cdf(self, x, logits=True, detach=False):
        x = x.transpose(0, 1).unsqueeze(-1)  # C, N, H, W, 1
        if detach:
            for i in range(self.chain_len - 1):
                x = self.affine[i](x, detach)
                x = x + torch.tanh(self.a[i].detach()) * torch.tanh(x)
            if logits:
                return self.affine[-1](x, detach).squeeze(-1).transpose(0, 1)
            return torch.sigmoid(self.affine[-1](x, detach)).squeeze(-1).transpose(0, 1)

        # not detached
        for i in range(self.chain_len - 1):
            x = self.affine[i](x)
            x = x + torch.tanh(self.a[i]) * torch.tanh(x)
        if logits:
            return self.affine[-1](x).squeeze(-1).transpose(0, 1)
        return torch.sigmoid(self.affine[-1](x)).squeeze(-1).transpose(0, 1)

    def pdf(self, x):
        cdf = self.cdf(x, False)
        jac = torch.ones_like(cdf)
        pdf = torch.autograd.grad(cdf, x, grad_outputs=jac)[0]
        return pdf

    def get_extraloss(self):
        target = 0
        logits = self.cdf(self._medians, detach=True)
        extra_loss = torch.abs(logits - target).sum()
        return extra_loss

    def likelihood(self, x, min=1e-9):
        lower = self.cdf(x - 0.5, True)
        upper = self.cdf(x + 0.5, True)
        sign = -torch.sign(lower + upper).detach()
        upper = torch.sigmoid(upper * sign)
        lower = torch.sigmoid(lower * sign)
        return LowerBound.apply(torch.abs(upper - lower), min)

    def icdf(self, xi, method='bisection', max_iterations=1000, tol=1e-9, **kwargs):
        if method == 'bisection':
            init_interval = [-1, 1]
            left_endpoints = torch.ones_like(xi) * init_interval[0]
            right_endpoints = torch.ones_like(xi) * init_interval[1]

            def f(z):
                return self.cdf(z, logits=False, detach=True) - xi

            while True:
                if (f(left_endpoints) < 0).all():
                    break
                else:
                    left_endpoints = left_endpoints * 2
            while True:
                if (f(right_endpoints) > 0).all():
                    break
                else:
                    right_endpoints = right_endpoints * 2

            for i in range(max_iterations):
                mid_pts = 0.5 * (left_endpoints + right_endpoints)
                mid_vals = f(mid_pts)
                pos = mid_vals > 0
                non_pos = torch.logical_not(pos)
                neg = mid_vals < 0
                non_neg = torch.logical_not(neg)
                left_endpoints = left_endpoints * non_neg.float() + mid_pts * neg.float()
                right_endpoints = right_endpoints * non_pos.float() + mid_pts * pos.float()
                if (torch.logical_and(non_pos, non_neg)).all() or torch.min(right_endpoints - left_endpoints) <= tol:
                    print(f'bisection terminated after {i} its')
                    break

            return mid_pts
        else:
            raise NotImplementedError

    def sample(self, img, shape):
        uni = torch.rand(shape, device=img.device)
        return self.icdf(uni)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_tensor(a, t, place_holder=None):
    return a[t, torch.arange(len(t))]


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps)


def noise(input, scale):
    return input + scale*(torch.rand_like(input) - 0.5)


def round_w_offset(input, loc):
    diff = STERound.apply(input - loc)
    return diff + loc


def quantize(x, mode='noise', offset=None):
    if mode == 'noise':
        return noise(x, 1)
    elif mode == 'round':
        return STERound.apply(x)
    elif mode == 'dequantize':
        return round_w_offset(x, offset)
    else:
        raise NotImplementedError


class STERound(Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class UpperBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.min(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs <= b
        pass_through_2 = grad_output > 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class NormalDistribution:
    '''
        A normal distribution
    '''
    def __init__(self, loc, scale):
        assert loc.shape == scale.shape
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        return self.loc.detach()

    def std_cdf(self, inputs):
        half = 0.5
        const = -(2**-0.5)
        return half * torch.erfc(const * inputs)

    def sample(self):
        return self.scale * torch.randn_like(self.scale) + self.loc

    def likelihood(self, x, min=1e-9):
        x = torch.abs(x - self.loc)
        upper = self.std_cdf((.5 - x) / self.scale)
        lower = self.std_cdf((-.5 - x) / self.scale)
        return LowerBound.apply(upper - lower, min)

    def scaled_likelihood(self, x, s=1, min=1e-9):
        x = torch.abs(x - self.loc)
        s = s * .5
        upper = self.std_cdf((s - x) / self.scale)
        lower = self.std_cdf((-s - x) / self.scale)
        return LowerBound.apply(upper - lower, min)