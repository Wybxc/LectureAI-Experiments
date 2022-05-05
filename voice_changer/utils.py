import asyncio
import nest_asyncio
import sys
import os
import subprocess
import tempfile
from io import BytesIO
from typing import Callable, Coroutine, Tuple

import librosa
import numpy as np
import soundfile
from pydub import AudioSegment
import functools

_iocp_loop = None


def _iocp_get_loop() -> asyncio.AbstractEventLoop:
    """获得 IOCP 事件循环，以修复在 Jupyter Notebook 里，Windows 系统没有使用 IOCP 的问题。

    Returns:
        asyncio.AbstractEventLoop: IOCP 的事件循环。
    """
    global _iocp_loop
    if sys.platform == "win32" and isinstance(
        asyncio.get_running_loop(), asyncio.SelectorEventLoop
    ):
        if not _iocp_loop:
            _iocp_loop = asyncio.ProactorEventLoop()  # 在 Windows 上使用 IOCP
            nest_asyncio.apply(_iocp_loop)
        return _iocp_loop
    return asyncio.get_running_loop()


async def _call_fut(func: Callable[..., Coroutine], fut: asyncio.Future):
    """调用异步函数，并将结果储存在 future 中。"""
    try:
        result = await func()
        fut.set_result(result)
    except Exception as e:
        fut.set_exception(e)


def _iocp_fix(func: Callable[..., Coroutine]):
    """修复在 Jupyter Notebook 里，Windows 系统没有使用 IOCP 的问题。"""

    try:
        loop = _iocp_get_loop()
        if _iocp_loop is not None:

            @functools.wraps(func)
            async def fixed(*args, **kwargs):
                fut = asyncio.get_running_loop().create_future()
                loop.run_until_complete(
                    _call_fut(functools.partial(func, *args, **kwargs), fut)
                )
                return await fut

            return fixed
    except:
        pass
    return func


def load_sound(filename: str) -> Tuple[np.ndarray, int]:
    """从文件中加载音频数据，自动识别格式，理论上支持 ffmpeg 的所有格式。"""
    sound: AudioSegment = AudioSegment.from_file(filename)
    wav_file: BytesIO = sound.export(BytesIO(), format="wav")
    return librosa.load(wav_file)


def encode_sound(data: np.ndarray, sr: int) -> AudioSegment:
    """将音频数据编码为 pydub 的 AudioSegment 对象。"""
    file = BytesIO()
    soundfile.write(file, data, sr, format="wav")
    return AudioSegment.from_wav(file)


def export_sound(data: np.ndarray, sr: int, filename: str, format: str = "") -> None:
    """将音频数据写入文件。"""
    sound = encode_sound(data, sr)
    if not format:
        parts = filename.split(".")
        if len(parts) > 1:
            format = parts[-1]
        else:
            format = "wav"
    sound.export(filename, format=format)


def play_sound(data: np.ndarray, sr: int) -> None:
    """播放音频数据，使用 ffplay。"""
    temp = tempfile.NamedTemporaryFile(mode="w+b", suffix=".wav", delete=False)
    temp.close()
    try:
        export_sound(data, sr, temp.name)
        proc = subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-hide_banner", temp.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip()
            raise RuntimeError(err)
    finally:
        os.remove(temp.name)


@_iocp_fix
async def play_sound_async(data: np.ndarray, sr: int):
    """播放音频数据（异步），使用 ffplay。"""
    temp = tempfile.NamedTemporaryFile(mode="w+b", suffix=".wav", delete=False)
    temp.close()
    try:
        export_sound(data, sr, temp.name)
        proc = await asyncio.subprocess.create_subprocess_exec(
            *("ffplay", "-nodisp", "-autoexit", "-hide_banner", temp.name),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.strip()
            raise RuntimeError(err)
    finally:
        os.remove(temp.name)
