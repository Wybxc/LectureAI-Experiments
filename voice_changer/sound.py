from typing import Any, Optional

import numpy as np

from .feature import (
    FFTConfig,
    formant_features,
    griffinlim,
    istft,
    melspectrogram,
    mfcc,
    phase_vocoder,
    resample,
    resample_to,
    stft,
    volume,
    yin,
    zero_cross,
)
from .utils import export_sound, load_sound, play_sound, play_sound_async


class Sound:
    """音频。"""

    y: np.ndarray
    """音频数据。"""
    sr: int
    """采样率。"""

    def __init__(self, y: np.ndarray, sr: int):
        self.y = y
        self.sr = sr

    @classmethod
    def from_file(cls, filename: str):
        """从文件中加载音频数据。"""
        y, sr = load_sound(filename)
        return cls(y, sr)

    def save(self, filename: str, format: str = ""):
        """保存音频数据到文件。"""
        export_sound(self.y, self.sr, filename, format)

    def play(self):
        """播放音频数据。"""
        play_sound(self.y, self.sr)

    async def play_async(self):
        """播放音频数据（异步）。"""
        await play_sound_async(self.y, self.sr)

    def display(self):
        from IPython.display import Audio, display

        display(Audio(data=self.y, rate=self.sr))

    def cut(self, start: float = 0, end: float = -1):
        """裁切出音频的一段（单位：秒）。"""
        start = int(start * self.sr)
        if start < 0:
            start = len(self) - start
        end = int(end * self.sr)
        if end < 0:
            end = len(self) - end
        return Sound(self.y[start:end], self.sr)

    def __len__(self):
        return len(self.y)

    def length(self) -> float:
        """音频长度（单位：秒）。"""
        return len(self) / self.sr

    def set_axis(self, ax: Any = None, fft: Optional[FFTConfig] = None):
        """设置绘图坐标轴。"""
        import matplotlib.pyplot as plt
        from .display import set_axis

        set_axis(ax or plt.gca(), self.sr, fft)

    def set_axis_by_frames(self, ax: Any = None, fft: Optional[FFTConfig] = None):
        """设置绘图坐标轴（分帧图）。"""
        import matplotlib.pyplot as plt
        from .display import set_axis_by_frames

        set_axis_by_frames(ax or plt.gca(), self.sr, fft)

    def stft(self, fft: Optional[FFTConfig] = None) -> "STFTMatrix":
        """STFT 频谱。"""
        return STFTMatrix(stft(self.y, fft))

    def melspectrogram(
        self, fft: Optional[FFTConfig] = None, n_mels: int = 32
    ) -> "MelSpectrogram":
        """Mel 频谱。"""
        return MelSpectrogram(melspectrogram(self.y, self.sr, fft, n_mels))

    def resample(self, rate: float):
        """重采样。"""
        return Sound(resample(self.y, rate, self.sr), self.sr)

    def resample_to(self, length: int):
        """重采样到指定长度。"""
        return Sound(resample_to(self.y, length), self.sr)

    def volume(self, fft: Optional[FFTConfig] = None) -> "Volume":
        """音量。"""
        return Volume(volume(self.y, fft))

    def zero_cross(self, fft: Optional[FFTConfig] = None) -> "ZeroCross":
        """过零率。"""
        return ZeroCross(zero_cross(self.y, fft))

    def yin(
        self,
        fft: Optional[FFTConfig] = None,
        f_min=80,
        f_max=800,
        pyin: bool = False,
    ):
        """基音频率。"""
        return F0(yin(self.y, self.sr, fft, f_min, f_max, pyin))

    def formant_features_from(
        self,
        stft_matrix: "STFTMatrix",
        f0: "F0",
        vol: "Volume",
        fft: Optional[FFTConfig] = None,
        n_features: int = 20,
    ) -> "FormantFeatures":
        """共振峰特性。"""
        return FormantFeatures(
            formant_features(
                stft_matrix.data, self.sr, f0.data, vol.data, fft, n_features
            )
        )

    def formant_features(
        self, fft: Optional[FFTConfig] = None, n_features: int = 20
    ) -> "FormantFeatures":
        """共振峰特性。"""
        return self.formant_features_from(
            self.stft(), self.yin(), self.volume(), fft, n_features
        )

    def pitch_shift(self, rate: float, fft: Optional[FFTConfig] = None):
        """变调。"""
        return (
            self.resample(1 / rate)
            .stft(fft)
            .phase_vocoder(1 / rate, fft)
            .istft(self.sr, fft)
        )


class ArrayContainer:
    data: np.ndarray

    def __init__(self, data: np.ndarray):
        self.data = data

    @property
    def shape(self):
        return self.data.shape


class SoundFrames(ArrayContainer):
    """音频的分帧特征（一维）。"""

    data: np.ndarray
    """音频的特征向量，大小 (n_frames)。"""

    def resample(self, rate: float, orig_sr: int = 22050):
        """重采样。"""
        return self.__class__(resample(self.data, rate, orig_sr))

    def resample_to(self, length: int):
        """重采样到指定长度。"""
        return self.__class__(resample_to(self.data, length))


class SoundFrames2d(ArrayContainer):
    """音频的分帧特征（二维）。"""

    data: np.ndarray
    """音频的特征矩阵，大小 (n_features, n_frames)。"""

    def resample(self, rate: float, orig_sr: int = 22050):
        """重采样。"""
        return self.__class__(resample(self.data, rate, orig_sr))

    def resample_to(self, length: int):
        """重采样到指定长度。"""
        return self.__class__(resample_to(self.data, length))


class SoundFramesComplex(ArrayContainer):
    """音频的分帧特征（二维复矩阵）。"""

    data: np.ndarray
    """音频的特征矩阵，复数，大小 (n_features, n_frames)。"""

    def resample(self, rate: float, orig_sr: int = 22050):
        """重采样。"""
        real = resample(self.data.real, rate, orig_sr)
        imag = resample(self.data.imag, rate, orig_sr)
        return self.__class__(real + 1j * imag)

    def resample_to(self, length: int):
        """重采样到指定长度。"""
        real = resample_to(self.data.real, length)
        imag = resample_to(self.data.imag, length)
        return self.__class__(real + 1j * imag)


class STFTMatrix(SoundFramesComplex):
    """STFT 复频谱矩阵。"""

    def istft(self, sr: int, fft: Optional[FFTConfig] = None) -> "Sound":
        return Sound(istft(self.data, fft), sr)

    def spectrogram(self) -> "STFTSpectrogram":
        return STFTSpectrogram(np.abs(self.data))

    def phase_vocoder(self, rate: float, fft: Optional[FFTConfig] = None):
        """相位声像器。"""
        return STFTMatrix(phase_vocoder(self.data, rate, fft=fft))


class MelSpectrogram(SoundFrames2d):
    """Mel 频谱。"""

    def mfcc(self, n_mfcc: int = 20) -> "MFCC":
        """MFCC 特征。"""
        return MFCC(mfcc(self.data, n_mfcc))


class STFTSpectrogram(SoundFrames2d):
    """STFT 频谱。"""

    def griffinlim(self, sr: int, fft: Optional[FFTConfig] = None) -> Sound:
        """Griffin-Lim 音频重建。"""
        return Sound(griffinlim(self.data, fft), sr)

    def apply_filter(self, filter_: np.ndarray):
        """应用滤波器。"""
        return self.__class__(self.data * filter_)


class Volume(SoundFrames):
    """相对音量。"""


class ZeroCross(SoundFrames):
    """过零率。"""


class F0(SoundFrames):
    """基音频率。"""


class MFCC(SoundFrames2d):
    """MFCC 特征。"""


class FormantFeatures(SoundFrames2d):
    """共振峰特性。"""
