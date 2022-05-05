from typing import Optional
import matplotlib.ticker as ticker
from matplotlib.axes import Axes

from .feature import FFTConfig, fft_default


def set_axis(ax: Axes, sr: int, fft: Optional[FFTConfig] = None):
    """设置坐标轴。"""
    fft = fft or fft_default

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.xaxis.set_ticks_position("top")
    ax.secondary_xaxis(
        "bottom", functions=(lambda x: x / sr, lambda x: x * sr)
    ).xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.yaxis.set_ticks_position("right")
    ax.secondary_yaxis(
        "left",
        functions=(lambda x: x / fft["n_fft"] * sr, lambda x: x * fft["n_fft"] / sr),
    ).yaxis.set_minor_locator(ticker.AutoMinorLocator(10))


def set_axis_by_frames(ax: Axes, sr: int, fft: Optional[FFTConfig] = None):
    """设置坐标轴（分帧图）。"""
    fft = fft or fft_default

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.xaxis.set_ticks_position("top")
    ax.secondary_xaxis(
        "bottom",
        functions=(
            lambda x: x * fft["hop_length"] / sr,
            lambda x: x / fft["hop_length"] * sr,
        ),
    ).xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.yaxis.set_ticks_position("right")
    ax.secondary_yaxis(
        "left",
        functions=(lambda x: x / fft["n_fft"] * sr, lambda x: x * fft["n_fft"] / sr),
    ).yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
