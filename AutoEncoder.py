import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from contextlib import contextmanager

# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

# from ldm.modules.diffusionmodules.model import Encoder, Decoder
# from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

# from ldm.util import instantiate_from_config


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


# class Encoder(nn.Module):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
#                  resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
#                  **ignore_kwargs):
#         super().__init__()
#         if use_linear_attn: attn_type = "linear"
#         self.ch = ch
#         self.temb_ch = 0
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels
#
#         # downsampling
#         self.conv_in = torch.nn.Conv2d(in_channels,
#                                        self.ch,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)
#
#         curr_res = resolution
#         in_ch_mult = (1,)+tuple(ch_mult)
#         self.in_ch_mult = in_ch_mult
#         self.down = nn.ModuleList()
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_in = ch*in_ch_mult[i_level]
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(make_attn(block_in, attn_type=attn_type))
#             down = nn.Module()
#             down.block = block
#             down.attn = attn
#             if i_level != self.num_resolutions-1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)
#
#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#
#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv2d(block_in,
#                                         2*z_channels if double_z else z_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)
#
#     def forward(self, x):
#         # timestep embedding
#         temb = None
#
#         # downsampling
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 if len(self.down[i_level].attn) > 0:
#                     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions-1:
#                 hs.append(self.down[i_level].downsample(hs[-1]))
#
#         # middle
#         h = hs[-1]
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)
#
#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h
#
#
# class Decoder(nn.Module):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
#                  resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
#                  attn_type="vanilla", **ignorekwargs):
#         super().__init__()
#         if use_linear_attn: attn_type = "linear"
#         self.ch = ch
#         self.temb_ch = 0
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels
#         self.give_pre_end = give_pre_end
#         self.tanh_out = tanh_out
#
#         # compute in_ch_mult, block_in and curr_res at lowest res
#         in_ch_mult = (1,)+tuple(ch_mult)
#         block_in = ch*ch_mult[self.num_resolutions-1]
#         curr_res = resolution // 2**(self.num_resolutions-1)
#         self.z_shape = (1,z_channels,curr_res,curr_res)
#         print("Working with z of shape {} = {} dimensions.".format(
#             self.z_shape, np.prod(self.z_shape)))
#
#         # z to block_in
#         self.conv_in = torch.nn.Conv2d(z_channels,
#                                        block_in,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)
#
#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#
#         # upsampling
#         self.up = nn.ModuleList()
#         for i_level in reversed(range(self.num_resolutions)):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks+1):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(make_attn(block_in, attn_type=attn_type))
#             up = nn.Module()
#             up.block = block
#             up.attn = attn
#             if i_level != 0:
#                 up.upsample = Upsample(block_in, resamp_with_conv)
#                 curr_res = curr_res * 2
#             self.up.insert(0, up) # prepend to get consistent order
#
#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv2d(block_in,
#                                         out_ch,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)
#
#     def forward(self, z):
#         #assert z.shape[1:] == self.z_shape[1:]
#         self.last_z_shape = z.shape
#
#         # timestep embedding
#         temb = None
#
#         # z to block_in
#         h = self.conv_in(z)
#
#         # middle
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)
#
#         # upsampling
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks+1):
#                 h = self.up[i_level].block[i_block](h, temb)
#                 if len(self.up[i_level].attn) > 0:
#                     h = self.up[i_level].attn[i_block](h)
#             if i_level != 0:
#                 h = self.up[i_level].upsample(h)
#
#         # end
#         if self.give_pre_end:
#             return h
#
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         if self.tanh_out:
#             h = torch.tanh(h)
#         return h


#
# class VQModel(pl.LightningModule):
#     def __init__(self,
#                  ddconfig,
#                  lossconfig,
#                  n_embed,
#                  embed_dim,
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  image_key="image",
#                  colorize_nlabels=None,
#                  monitor=None,
#                  batch_resize_range=None,
#                  scheduler_config=None,
#                  lr_g_factor=1.0,
#                  remap=None,
#                  sane_index_shape=False, # tell vector quantizer to return indices as bhw
#                  use_ema=False
#                  ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.n_embed = n_embed
#         self.image_key = image_key
#         self.encoder = Encoder(**ddconfig)
#         self.decoder = Decoder(**ddconfig)
#         self.loss = instantiate_from_config(lossconfig)
#         self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
#                                         remap=remap,
#                                         sane_index_shape=sane_index_shape)
#         self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
#         self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
#         if colorize_nlabels is not None:
#             assert type(colorize_nlabels)==int
#             self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
#         if monitor is not None:
#             self.monitor = monitor
#         self.batch_resize_range = batch_resize_range
#         if self.batch_resize_range is not None:
#             print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")
#
#         self.use_ema = use_ema
#         if self.use_ema:
#             self.model_ema = LitEma(self)
#             print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
#
#         if ckpt_path is not None:
#             self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
#         self.scheduler_config = scheduler_config
#         self.lr_g_factor = lr_g_factor
#
#     @contextmanager
#     def ema_scope(self, context=None):
#         if self.use_ema:
#             self.model_ema.store(self.parameters())
#             self.model_ema.copy_to(self)
#             if context is not None:
#                 print(f"{context}: Switched to EMA weights")
#         try:
#             yield None
#         finally:
#             if self.use_ema:
#                 self.model_ema.restore(self.parameters())
#                 if context is not None:
#                     print(f"{context}: Restored training weights")
#
#     def init_from_ckpt(self, path, ignore_keys=list()):
#         sd = torch.load(path, map_location="cpu")["state_dict"]
#         keys = list(sd.keys())
#         for k in keys:
#             for ik in ignore_keys:
#                 if k.startswith(ik):
#                     print("Deleting key {} from state_dict.".format(k))
#                     del sd[k]
#         missing, unexpected = self.load_state_dict(sd, strict=False)
#         print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
#         if len(missing) > 0:
#             print(f"Missing Keys: {missing}")
#             print(f"Unexpected Keys: {unexpected}")
#
#     def on_train_batch_end(self, *args, **kwargs):
#         if self.use_ema:
#             self.model_ema(self)
#
#     def encode(self, x):
#         h = self.encoder(x)
#         h = self.quant_conv(h)
#         quant, emb_loss, info = self.quantize(h)
#         return quant, emb_loss, info
#
#     def encode_to_prequant(self, x):
#         h = self.encoder(x)
#         h = self.quant_conv(h)
#         return h
#
#     def decode(self, quant):
#         quant = self.post_quant_conv(quant)
#         dec = self.decoder(quant)
#         return dec
#
#     def decode_code(self, code_b):
#         quant_b = self.quantize.embed_code(code_b)
#         dec = self.decode(quant_b)
#         return dec
#
#     def forward(self, input, return_pred_indices=False):
#         quant, diff, (_,_,ind) = self.encode(input)
#         dec = self.decode(quant)
#         if return_pred_indices:
#             return dec, diff, ind
#         return dec, diff
#
#     def get_input(self, batch, k):
#         x = batch[k]
#         if len(x.shape) == 3:
#             x = x[..., None]
#         x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
#         if self.batch_resize_range is not None:
#             lower_size = self.batch_resize_range[0]
#             upper_size = self.batch_resize_range[1]
#             if self.global_step <= 4:
#                 # do the first few batches with max size to avoid later oom
#                 new_resize = upper_size
#             else:
#                 new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
#             if new_resize != x.shape[2]:
#                 x = F.interpolate(x, size=new_resize, mode="bicubic")
#             x = x.detach()
#         return x
#
#     def training_step(self, batch, batch_idx, optimizer_idx):
#         # https://github.com/pytorch/pytorch/issues/37142
#         # try not to fool the heuristics
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss, ind = self(x, return_pred_indices=True)
#
#         if optimizer_idx == 0:
#             # autoencode
#             aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
#                                             last_layer=self.get_last_layer(), split="train",
#                                             predicted_indices=ind)
#
#             self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#             return aeloss
#
#         if optimizer_idx == 1:
#             # discriminator
#             discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
#                                             last_layer=self.get_last_layer(), split="train")
#             self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#             return discloss
#
#     def validation_step(self, batch, batch_idx):
#         log_dict = self._validation_step(batch, batch_idx)
#         with self.ema_scope():
#             log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
#         return log_dict
#
#     def _validation_step(self, batch, batch_idx, suffix=""):
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss, ind = self(x, return_pred_indices=True)
#         aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
#                                         self.global_step,
#                                         last_layer=self.get_last_layer(),
#                                         split="val"+suffix,
#                                         predicted_indices=ind
#                                         )
#
#         discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
#                                             self.global_step,
#                                             last_layer=self.get_last_layer(),
#                                             split="val"+suffix,
#                                             predicted_indices=ind
#                                             )
#         rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
#         self.log(f"val{suffix}/rec_loss", rec_loss,
#                    prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
#         self.log(f"val{suffix}/aeloss", aeloss,
#                    prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
#         if version.parse(pl.__version__) >= version.parse('1.4.0'):
#             del log_dict_ae[f"val{suffix}/rec_loss"]
#         self.log_dict(log_dict_ae)
#         self.log_dict(log_dict_disc)
#         return self.log_dict
#
#     def configure_optimizers(self):
#         lr_d = self.learning_rate
#         lr_g = self.lr_g_factor*self.learning_rate
#         print("lr_d", lr_d)
#         print("lr_g", lr_g)
#         opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
#                                   list(self.decoder.parameters())+
#                                   list(self.quantize.parameters())+
#                                   list(self.quant_conv.parameters())+
#                                   list(self.post_quant_conv.parameters()),
#                                   lr=lr_g, betas=(0.5, 0.9))
#         opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
#                                     lr=lr_d, betas=(0.5, 0.9))
#
#         if self.scheduler_config is not None:
#             scheduler = instantiate_from_config(self.scheduler_config)
#
#             print("Setting up LambdaLR scheduler...")
#             scheduler = [
#                 {
#                     'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
#                     'interval': 'step',
#                     'frequency': 1
#                 },
#                 {
#                     'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
#                     'interval': 'step',
#                     'frequency': 1
#                 },
#             ]
#             return [opt_ae, opt_disc], scheduler
#         return [opt_ae, opt_disc], []
#
#     def get_last_layer(self):
#         return self.decoder.conv_out.weight
#
#     def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
#         log = dict()
#         x = self.get_input(batch, self.image_key)
#         x = x.to(self.device)
#         if only_inputs:
#             log["inputs"] = x
#             return log
#         xrec, _ = self(x)
#         if x.shape[1] > 3:
#             # colorize with random projection
#             assert xrec.shape[1] > 3
#             x = self.to_rgb(x)
#             xrec = self.to_rgb(xrec)
#         log["inputs"] = x
#         log["reconstructions"] = xrec
#         if plot_ema:
#             with self.ema_scope():
#                 xrec_ema, _ = self(x)
#                 if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
#                 log["reconstructions_ema"] = xrec_ema
#         return log
#
#     def to_rgb(self, x):
#         assert self.image_key == "segmentation"
#         if not hasattr(self, "colorize"):
#             self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
#         x = F.conv2d(x, weight=self.colorize)
#         x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
#         return x
#
#
# class VQModelInterface(VQModel):
#     def __init__(self, embed_dim, *args, **kwargs):
#         super().__init__(embed_dim=embed_dim, *args, **kwargs)
#         self.embed_dim = embed_dim
#
#     def encode(self, x):
#         h = self.encoder(x)
#         h = self.quant_conv(h)
#         return h
#
#     def decode(self, h, force_not_quantize=False):
#         # also go through quantization layer
#         if not force_not_quantize:
#             quant, emb_loss, info = self.quantize(h)
#         else:
#             quant = h
#         quant = self.post_quant_conv(quant)
#         dec = self.decoder(quant)
#         return dec


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


# class IdentityFirstStage(torch.nn.Module):
#     def __init__(self, *args, vq_interface=False, **kwargs):
#         self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
#         super().__init__()
#
#     def encode(self, x, *args, **kwargs):
#         return x
#
#     def decode(self, x, *args, **kwargs):
#         return x
#
#     def quantize(self, x, *args, **kwargs):
#         if self.vq_interface:
#             return x, None, [None, None, None]
#         return x
#
#     def forward(self, x, *args, **kwargs):
#         return x



