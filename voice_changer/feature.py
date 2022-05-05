from typing import Literal, Optional, TypedDict

import librosa
import scipy.signal as signal
import numpy as np


class FFTConfig(TypedDict):
    """FFT 参数设置。"""

    n_fft: int
    """FFT 点数。"""
    win_length: int
    """窗长。"""
    hop_length: int
    """帧长。"""
    window: Literal[
        "boxcar",
        "triang",
        "blackman",
        "hamming",
        "hann",
        "bartlett",
        "flattop",
        "parzen",
        "bohman",
        "blackmanharris",
        "nuttall",
        "barthann",
        "cosine",
        "exponential",
        "tukey",
        "taylor",
    ]
    """加窗类型。"""


fft_default: FFTConfig = {
    "n_fft": 512,
    "win_length": 512,
    "hop_length": 128,
    "window": "cosine",
}
"""默认 FFT 配置。"""


def stft(y: np.ndarray, fft: Optional[FFTConfig] = None):
    """对音频进行 STFT 变换。"""
    fft = fft or fft_default
    return librosa.stft(y, **fft)


def istft(stft_matrix: np.ndarray, fft: Optional[FFTConfig] = None):
    """STFT 逆变换。"""
    fft = fft or fft_default
    return librosa.istft(stft_matrix, **fft)


def melspectrogram(
    y: np.ndarray, sr: int, fft: Optional[FFTConfig] = None, n_mels: int = 128
):
    """计算音频的 Mel 频率。"""
    fft = fft or fft_default
    return librosa.feature.melspectrogram(y=y, sr=sr, **fft, n_mels=n_mels)


def griffinlim(S: np.ndarray, fft: Optional[FFTConfig] = None):
    """Griffin-lim 音频信号重建。"""
    fft = fft or fft_default
    return librosa.griffinlim(S, **fft)


def resample(y: np.ndarray, rate: float, orig_sr: int = 22050):
    """重采样。"""
    return librosa.resample(y, orig_sr=orig_sr, target_sr=orig_sr * rate)


def resample_to(y: np.ndarray, length: int):
    """重采样到指定长度。"""
    if len(y.shape) == 1:
        return signal.resample(y, length)
    else:
        return signal.resample(y, length, axis=1)


def phase_vocoder(
    stft_matrix: np.ndarray, rate: float, fft: Optional[FFTConfig] = None
):
    """相位声像器。"""
    fft = fft or fft_default
    return librosa.phase_vocoder(stft_matrix, rate=rate, hop_length=fft["hop_length"])


def volume(y: np.ndarray, fft: Optional[FFTConfig] = None):
    """计算音频的分帧相对音量。"""
    fft = fft or fft_default
    result = librosa.feature.rms(
        y=y, frame_length=fft["win_length"], hop_length=fft["hop_length"]
    ).reshape(-1)
    return result / np.max(result)


def zero_cross(y: np.ndarray, fft: Optional[FFTConfig] = None):
    """计算音频的分帧过零率。"""
    fft = fft or fft_default
    return librosa.feature.zero_crossing_rate(
        y=y, frame_length=fft["win_length"], hop_length=fft["hop_length"]
    ).reshape(-1)


def yin(
    y: np.ndarray,
    sr: int,
    fft: Optional[FFTConfig] = None,
    f_min=80,
    f_max=800,
    pyin: bool = False,
):
    """寻找基频。

    参数 pyin 指定是否使用 pyin 算法，pyin 准确度更高，但速度慢。
    """
    fft = fft or fft_default
    if not pyin:
        return librosa.yin(
            y=y,
            sr=sr,
            fmin=f_min,
            fmax=f_max,
            frame_length=fft["win_length"],
            hop_length=fft["hop_length"],
        )
    else:
        f0, _, _ = librosa.pyin(
            y=y,
            sr=sr,
            fmin=f_min,
            fmax=f_max,
            frame_length=fft["win_length"],
            hop_length=fft["hop_length"],
        )
        return f0


def formant_features(
    stft_matrix: np.ndarray,
    sr: int,
    f0: np.ndarray,
    vol: np.ndarray,
    fft: Optional[FFTConfig] = None,
    n_features: int = 20,
):
    """频域特性：各个共振峰的相对能量。

    Args:
        stft_matrix: STFT 矩阵。
        sr: 采样率。
        f0: 基频。
        vol: 音量。
        fft: FFT 配置。
        n_features: 特征数量。
    """
    fft = fft or fft_default
    nsf = np.abs(stft_matrix)  # 幅度谱

    fq_fea = np.zeros((n_features, stft_matrix.shape[1]), dtype=np.float32)
    for i in range(1, n_features + 1):
        freq_l = np.floor(f0 * fft["n_fft"] / sr * (i - 0.5)).astype(np.int32)
        freq_h = np.floor(f0 * fft["n_fft"] / sr * (i + 0.5)).astype(np.int32)
        for j in range(stft_matrix.shape[1]):
            fq_fea[i - 1, j] = np.sqrt(np.sum(nsf[freq_l[j] : freq_h[j], j] ** 2))
    np.nan_to_num(fq_fea, copy=False)
    np.divide(fq_fea, vol, out=fq_fea, where=vol > 0.2)  # 去除音量的影响
    fq_fea /= fq_fea.max()  # 归一化
    return fq_fea


def mfcc(mel_matrix: np.ndarray, n_mfcc: int = 20):
    """计算音频的 MFCC 特征。"""
    return librosa.feature.mfcc(S=mel_matrix, n_mfcc=n_mfcc)
