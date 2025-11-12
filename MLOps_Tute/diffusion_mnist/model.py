import math
import torch
import torch.nn as nn
import lightning as L

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


# -- Position embeddings
class SinusoidalPosEmb(nn.Module):
    # TODO: Investigate
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


# -- U-net blocks based on https://arxiv.org/pdf/1505.04597
class DoubleConv(nn.Module):
    """
    Double Convolution block with slight activation func deviation
    (convolution => [Norm] => Mish) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 n_groups=1):
        super().__init__()

        # Determine convolutional channels for mid
        if not mid_channels:
            mid_channels = out_channels

        # Double convolution block
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=mid_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=n_groups,
                         num_channels=mid_channels),
            nn.Mish(),      # Deviation from Relu
            nn.Conv2d(in_channels=mid_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
            nn.Mish(),      # Deviation from Relu
        )

    def forward(self, x):
        return self.block(x)


class DownMaxPoolConv(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 n_groups=1, time_emb_dim=256):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels,
                       mid_channels=mid_channels, n_groups=n_groups)
        )

        # Time-embedding layer
        self.time_emb_layer = nn.Sequential(
            nn.Mish(),
            nn.Linear(
                time_emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t=None):
        if t is not None:
            x = self.maxpool_conv(x)
            # Broadcast time embedding to (B, C, H, W)
            # t = t[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            t = self.time_emb_layer(t)[:, :, None, None].repeat(
                1, 1, x.shape[-2], x.shape[-1])
            return x + t
        else:
            return self.maxpool_conv(x)


class UpConv(nn.Module):
    """
    Upscaling with conv transpose then double conv
    """
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 n_groups=1, time_emb_dim=256):
        super().__init__()
        # Upsample with conv transpose but have number of channels halve
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=in_channels // 2,
                               kernel_size=2,
                               stride=2),
            DoubleConv(in_channels=in_channels // 2,
                       out_channels=out_channels,
                       mid_channels=mid_channels,
                       n_groups=n_groups)
        )

        # Individual Up Sample blocks + Double conv
        self.up = nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=in_channels // 2,
                                     kernel_size=4,
                                     stride=2,
                                     padding=1)
        self.double_conv = DoubleConv(in_channels=in_channels,
                                      out_channels=out_channels,
                                      mid_channels=mid_channels,
                                      n_groups=n_groups)

        # Time-embedding layer
        self.time_emb_layer = nn.Sequential(
            nn.Mish(),
            nn.Linear(
                time_emb_dim,
                out_channels
            ),
        )

    def forward(self, x1, x2=None, t=None):
        if x2 is not None:
            # -- Deal with residual
            # Input is of (Batch size, channels, height, width)
            # TODO: Deal with padding

            # Upscale first
            x1 = self.up(x1)

            # Concat along channels axis
            x = torch.cat([x2, x1], dim=1)

            x = self.double_conv(x)
        else:
            # Otherwise up scale and double conv
            x = self.up_conv(x1)

        # Deal with time-embedding
        if t is not None:
            # t = t[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            t = self.time_emb_layer(t)[:, :, None, None].repeat(
                1, 1, x.shape[-2], x.shape[-1])
            return x + t
        else:
            return x


# TODO:
# Look at self-attention implementation for proper implementation
# https://docs.pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html#vit_b_16
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # End x: (-1, self.size*self.size, self.channels)
        # Usually its x: (seq_len, batch_size, input_dim)
        # query inputs are: (L, N, Eq) --> (seq_len, batch_size, emb_dim)
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        # self.mha(query, key, value)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(
            -1, self.channels, self.size, self.size)


# -- U-Net based on https://arxiv.org/pdf/1505.04597
class UNetClassic(nn.Module):
    """
    U-Net classic
    Based on architecture specified in https://arxiv.org/pdf/1505.04597 (fig1)
    Implementation details:
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    """
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()

        self.device = device
        self.time_dim = time_dim    # Time diffusion step embedding dimension

        # -- Diffusion step encoder
        # Time embeddings
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dim=self.time_dim),
            # nn.Linear(self.time_dim, self.time_dim*4),
            # nn.Mish(),
            # nn.Linear(self.time_dim*4, self.time_dim),
        )
        self.diffusion_step_encoder = diffusion_step_encoder

        # -- U-net blocks
        self.inc = DoubleConv(in_channels=c_in, out_channels=64)

        # (H,W) = (14,14)
        self.down1 = DownMaxPoolConv(in_channels=64, out_channels=128)
        # (H,W) = (7,7)
        self.down2 = DownMaxPoolConv(in_channels=128, out_channels=256)

        self.up1 = UpConv(in_channels=256, out_channels=128)
        self.up2 = UpConv(in_channels=128, out_channels=64)
        self.outc = nn.Conv2d(in_channels=64, out_channels=c_out,
                              kernel_size=3, padding=1)

    def forward(self, sample, timestep):
        # -- Time Embeddings
        # Get timestep embeddings
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass
            # timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long,
                                     device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way
        # that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.diffusion_step_encoder(timesteps)

        # -- Sample Forward pass
        x1 = self.inc(sample)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)

        x = self.up1(x3, x2, t_emb)
        x = self.up2(x, x1, t_emb)
        logits = self.outc(x)
        return logits


# -- Lightning system
# Refering to style guide specified:
# https://lightning.ai/docs/pytorch/stable/starter/style_guide.html
class UNetDiffSystem(L.LightningModule):
    def __init__(self, num_diffusion_iters: int = 100,
                 learning_rate: float = 1e-4):
        super().__init__()

        # -- Params
        self.lr = learning_rate

        # -- Model init
        self.unet = UNetClassic()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

    def forward(self, sample, timestep):
        return self.unet(sample, timestep)

    def sample(self):
        # Set to eval mode for sampling
        self.unet.eval()

        with torch.no_grad():
            # Initial noisy image
            nimage = torch.randn((1, 1, 28, 28)).to(self.device)

            # init scheduler
            if self.noise_scheduler.num_inference_steps is None:
                num_inference_steps = \
                    self.noise_scheduler.config.num_train_timesteps
            else:
                num_inference_steps = \
                    self.noise_scheduler.num_inference_steps

            self.noise_scheduler.set_timesteps(
                num_inference_steps)

            # Diffusion sampling
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.forward(
                    sample=nimage,
                    timestep=k,
                )

                # inverse diffusion step (remove noise)
                nimage = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=nimage
                ).prev_sample

        # Reset to train mode
        self.unet.train()

        return nimage

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, _ = batch

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (images.shape[0],), device=self.device
        ).to(device=self.device)

        # sample noise to add to image
        noise = torch.randn(images.shape, device=self.device)

        # add noise to the clean images according to the noise magnitude
        # at each diffusion iteration (this is the forward diffusion process)
        noisy_image = self.noise_scheduler.add_noise(
            images, noise, timesteps)

        # predict noise
        predicted_noise = self.forward(sample=noisy_image, timestep=timesteps)
        loss = nn.MSELoss()(predicted_noise, noise)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    # def on_train_epoch_end(self):
    #     # images = wandb
    #     image = self.sample()
    #     image = torch.permute(image[0], (1, 2, 0)).detach().cpu().numpy()

    #     # Assumes wandb logger
    #     self.logger.log_image(key="samples", images=[image],
    #                           step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        images, _ = batch

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (images.shape[0],), device=self.device
        ).to(device=self.device)

        # sample noise to add to image
        noise = torch.randn(images.shape, device=self.device)

        # add noise to the clean images according to the noise magnitude
        # at each diffusion iteration (this is the forward diffusion process)
        noisy_image = self.noise_scheduler.add_noise(
            images, noise, timesteps)

        # predict noise
        predicted_noise = self.forward(sample=noisy_image, timestep=timesteps)
        loss = nn.MSELoss()(predicted_noise, noise)

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)

    def on_validation_epoch_end(self):
        image = self.sample()
        image = torch.permute(image[0], (1, 2, 0)).detach().cpu().numpy()

        # Assumes wandb logger
        self.logger.log_image(key="samples", images=[image],
                              step=self.current_epoch)
