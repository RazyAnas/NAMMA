"""Pure-PyTorch fallback for gsplat's fused_ssim import.

This slower fallback keeps the dedicated panorama job independent of the optional
native fused_ssim CUDA extension. The function signature matches gsplat v1.5.3.
"""
import torch
import torch.nn.functional as F


def _window(channels, device, dtype):
    coords = torch.arange(11, device=device, dtype=dtype) - 5
    one = torch.exp(-(coords * coords) / (2 * 1.5 ** 2))
    one = one / one.sum()
    window = one[:, None] @ one[None, :]
    return window.expand(channels, 1, 11, 11).contiguous()


def fused_ssim(img1, img2, padding="same", train=True):
    channels = img1.shape[1]
    window = _window(channels, img1.device, img1.dtype)
    pad = 5 if padding == "same" else 0
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    sigma1 = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1.square()
    sigma2 = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2.square()
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1.square() + mu2.square() + c1) * (sigma1 + sigma2 + c2)
    return (numerator / denominator).mean()
