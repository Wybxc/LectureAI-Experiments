from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from .feature import FFTConfig, fft_default
from .sound import Sound


class Model:
    """模型。"""

    def fit(self, *args, **kwargs):
        """拟合。"""
        return self

    def predict(self, *args, **kwargs):
        """应用。"""

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class PitchChanger(Model):
    """变调器。"""

    def __init__(self):
        self._rate = 1.0

    def fit(self, source: Sound, dest: Sound):
        f0_src = np.nanmean(source.yin(pyin=True).data)
        f0_dst = np.nanmean(dest.yin(pyin=True).data)
        self._rate = f0_dst / f0_src
        return self

    def predict(self, source: Sound) -> Sound:
        return source.pitch_shift(self._rate)


class MonoFilter(Model):
    """单音节滤波器。"""

    def __init__(self, n_features: int = 20, fft: Optional[FFTConfig] = None):
        self.n_features = n_features
        self.fft = fft or fft_default
        self._cov = np.zeros(n_features)
        self._sov = np.zeros(n_features)
        self._filter = np.ones(n_features)

    def fit(self, source: Sound, dest: Sound):
        # 计算共振峰特性
        ffea_src = source.formant_features(n_features=self.n_features)
        ffea_dst = dest.formant_features(n_features=self.n_features)

        # 重采样两组特征到相同长度
        ffea_src = ffea_src.resample_to(1024)
        ffea_dst = ffea_dst.resample_to(1024)

        # 最小二乘法估计滤波器
        self._cov += np.einsum("ij,ij->i", ffea_src.data, ffea_dst.data)  # 累积计算
        self._sov += np.einsum("ij,ij->i", ffea_src.data, ffea_src.data)
        self._filter = self._cov / self._sov

        return self

    def predict(self, source: Sound) -> Sound:
        f0_src = source.yin()
        stft_src = source.stft()
        n_stft, length = stft_src.shape

        fx = np.arange(1, self.n_features + 1).reshape(-1, 1) * f0_src.data
        fx *= self.fft["n_fft"] / source.sr

        ff = np.empty((n_stft, length))
        for i in range(length):
            interp = interp1d(
                fx[:, i],
                self._filter,
                bounds_error=False,
                fill_value=(1, 1),
                kind="cubic",
            )
            ff[:, i] = interp(np.arange(n_stft))
        
        # 滑窗平均
        for i in range(n_stft):
            ff[i] = np.convolve(ff[i], np.ones(10) / 10, mode="same")

        return stft_src.spectrogram().apply_filter(ff).griffinlim(source.sr, self.fft)
