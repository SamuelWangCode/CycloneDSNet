# typhoon_intensity_bc/model/dsnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from typhoon_intensity_bc.model.base_method import Base_method
from typhoon_intensity_bc.model.model import BCDownscaleModel


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda: window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1.0 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        self.kernel_x = torch.tensor(kernel_x).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.tensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        if pred.device != self.kernel_x.device:
            self.kernel_x = self.kernel_x.to(pred.device)
            self.kernel_y = self.kernel_y.to(pred.device)
        loss = 0
        channels = pred.shape[1]
        for c in range(channels):
            p_c = pred[:, c:c + 1]
            t_c = target[:, c:c + 1]
            grad_x_p = F.conv2d(p_c, self.kernel_x, padding=1)
            grad_y_p = F.conv2d(p_c, self.kernel_y, padding=1)
            grad_x_t = F.conv2d(t_c, self.kernel_x, padding=1)
            grad_y_t = F.conv2d(t_c, self.kernel_y, padding=1)
            loss += self.l1(torch.abs(grad_x_p) + torch.abs(grad_y_p), torch.abs(grad_x_t) + torch.abs(grad_y_t))
        return loss / channels


class PhysicalIntensityLoss(nn.Module):
    def __init__(self, normalizer_mean, normalizer_std, mslp_idx=3, u_idx=0, v_idx=1, weight=1.0, img_size=131):
        super().__init__()
        self.register_buffer('mean', normalizer_mean)
        self.register_buffer('std', normalizer_std)
        self.mslp_idx = mslp_idx
        self.u_idx = u_idx
        self.v_idx = v_idx
        self.weight = weight
        self.l1 = nn.L1Loss()

        y = torch.arange(img_size).view(1, -1, 1).float()
        x = torch.arange(img_size).view(1, 1, -1).float()
        self.register_buffer('grid_y', y)
        self.register_buffer('grid_x', x)
        self.img_size = img_size

    def _generate_simple_mask(self, target_phys):
        B, _, H, W = target_phys.shape
        mslp = target_phys[:, self.mslp_idx, :, :]
        flat_mslp = mslp.view(B, -1)
        _, min_indices = torch.min(flat_mslp, dim=1)
        center_y = (min_indices // W).view(B, 1, 1, 1).float()
        center_x = (min_indices % W).view(B, 1, 1, 1).float()
        dist_sq = (self.grid_x - center_x) ** 2 + (self.grid_y - center_y) ** 2
        dist_norm = torch.sqrt(dist_sq) / (self.img_size * 0.5)

        mask = torch.zeros_like(dist_norm)
        mask[(dist_norm > 0.1) & (dist_norm < 0.6)] = 1.0
        return mask

    def forward(self, pred_field, target_field, cma_pres_true, cma_wind_true):
        pred_phys = pred_field * self.std + self.mean
        target_phys = target_field * self.std + self.mean

        mask = self._generate_simple_mask(target_phys)

        pred_u = pred_phys[:, self.u_idx, :, :]
        pred_v = pred_phys[:, self.v_idx, :, :]
        pred_ws = torch.sqrt(pred_u ** 2 + pred_v ** 2 + 1e-6)
        pred_mslp = pred_phys[:, self.mslp_idx, :, :]

        masked_ws = pred_ws * mask.squeeze(1)
        flat_ws = masked_ws.view(masked_ws.size(0), -1)
        topk_ws_vals, _ = torch.topk(flat_ws, 20, dim=1)
        pred_max_wind = topk_ws_vals.mean(dim=1)

        flat_mslp = pred_mslp.view(pred_mslp.size(0), -1)
        topk_mslp_vals, _ = torch.topk(-flat_mslp, 20, dim=1)
        pred_min_pres = -topk_mslp_vals.mean(dim=1)

        target_wind_relaxed = cma_wind_true * 0.92

        loss_p = self.l1(pred_min_pres, cma_pres_true) * 0.05
        loss_w = self.l1(pred_max_wind, target_wind_relaxed) * 1.0

        return (loss_p + loss_w) * self.weight


# === 3. DSNet 模型 ===
class DSNet(Base_method):
    def __init__(self, normalizer_stats=None, **kwargs):
        super().__init__(**kwargs)
        model_conf = kwargs.get('model_config', {})
        current_img_size = model_conf.get('hr_size', (131, 131))[0]
        print(f"[DSNet Init] Detected HR Image Size: {current_img_size}x{current_img_size}")

        self.pixel_loss = nn.L1Loss()
        self.grad_loss = GradientLoss()
        self.ssim_loss = SSIMLoss()

        if normalizer_stats is not None:
            self.int_loss = PhysicalIntensityLoss(
                normalizer_mean=normalizer_stats['mean'],
                normalizer_std=normalizer_stats['std'],
                u_idx=0, v_idx=1, mslp_idx=3,
                weight=1.0,
                img_size=current_img_size
            )
        else:
            self.int_loss = None

        self.skip_proj = nn.Conv2d(4, 4, kernel_size=1, bias=True, padding_mode='zeros')
        nn.init.dirac_(self.skip_proj.weight)
        if self.skip_proj.bias is not None: nn.init.zeros_(self.skip_proj.bias)

    def _build_model(self, **args):
        return BCDownscaleModel(**self.hparams.model_config)

    def forward(self, x1, x2, **kwargs):
        seq_out = self.model(x1, x2)
        residual = seq_out.squeeze(1)

        target_indices = [0, 1, 4, 6]
        input_base_lr = x1[:, -1, target_indices, :, :]
        input_base_mapped = self.skip_proj(input_base_lr)
        target_size = residual.shape[-2:]
        input_base_hr = F.interpolate(input_base_mapped, size=target_size, mode='bilinear', align_corners=False)
        final_out = residual + input_base_hr
        return final_out

    def training_step(self, batch, batch_idx):
        batch_x1, batch_x2, batch_y, batch_cma = batch
        cma_pres, cma_wind = batch_cma
        pred_y = self(batch_x1, batch_x2)

        loss_pixel_u = F.l1_loss(pred_y[:, 0], batch_y[:, 0])
        loss_pixel_v = F.l1_loss(pred_y[:, 1], batch_y[:, 1])
        loss_pixel_t = F.l1_loss(pred_y[:, 2], batch_y[:, 2])
        loss_pixel_p = F.l1_loss(pred_y[:, 3], batch_y[:, 3])

        loss_pixel_weighted = (1.0 * loss_pixel_u + 1.0 * loss_pixel_v +
                               0.1 * loss_pixel_t + 5.0 * loss_pixel_p) / 4.0
        loss_grad = self.grad_loss(pred_y, batch_y)
        loss_ssim = self.ssim_loss(pred_y, batch_y)

        loss_int = 0.0
        if self.int_loss is not None:
            loss_int = self.int_loss(pred_y, batch_y, cma_pres, cma_wind)

        total_loss = 20.0 * loss_pixel_weighted + 10.0 * loss_ssim + 5.0 * loss_grad + 0.005 * loss_int

        self.log_dict({
            'train_loss': total_loss,
            'l_pix': loss_pixel_weighted, 'l_grad': loss_grad,
            'l_ssim': loss_ssim, 'l_int': loss_int
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        batch_x1, batch_x2, batch_y, batch_cma = batch
        cma_pres, cma_wind = batch_cma
        pred_y = self(batch_x1, batch_x2)

        loss_pixel_u = F.l1_loss(pred_y[:, 0], batch_y[:, 0])
        loss_pixel_v = F.l1_loss(pred_y[:, 1], batch_y[:, 1])
        loss_pixel_t = F.l1_loss(pred_y[:, 2], batch_y[:, 2])
        loss_pixel_p = F.l1_loss(pred_y[:, 3], batch_y[:, 3])

        loss_pixel_weighted = (1.0 * loss_pixel_u + 1.0 * loss_pixel_v +
                               0.1 * loss_pixel_t + 5.0 * loss_pixel_p) / 4.0
        loss_grad = self.grad_loss(pred_y, batch_y)
        loss_ssim = self.ssim_loss(pred_y, batch_y)
        loss_int = 0.0
        if self.int_loss is not None:
            loss_int = self.int_loss(pred_y, batch_y, cma_pres, cma_wind)

        val_loss = 20.0 * loss_pixel_weighted + 10.0 * loss_ssim + 5.0 * loss_grad + 0.005 * loss_int
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_pix', loss_pixel_weighted, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        batch_x1, batch_x2, batch_y, _ = batch
        pred_y = self(batch_x1, batch_x2)
        loss = self.pixel_loss(pred_y, batch_y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
