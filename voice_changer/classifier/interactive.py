from collections import Counter, deque
from matplotlib.figure import Figure

import numpy as np
import sounddevice as sd

from ..sound import Sound
from .data import MFCCData
from .model import MLPClassifier
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def record(
    classifier: MLPClassifier,
    data: MFCCData,
) -> None:
    """录音并实时分析。"""
    sr = 16000

    root = tk.Tk()
    root.title("实时语音分析")
    lbl_name = tk.Label(root, text="-")
    lbl_name.pack()

    figure = Figure()
    canvas = FigureCanvasTkAgg(figure, root)
    # 用draw代替
    canvas.draw()
    canvas.get_tk_widget().pack()

    data_history = deque()

    def callback_wrapper(indata: np.ndarray, frames: int, time, status) -> None:
        data_history.append(indata.reshape(-1).copy() * 10)
        if len(data_history) > 15:
            data_history.popleft()
        indata = np.concatenate(data_history)
        snd = Sound(indata, sr)
        mel = snd.melspectrogram()
        mfcc = mel.mfcc()
        names = Counter(classifier.analyze(mfcc, data)).most_common()
        lbl_name.configure(text=", ".join(name for name, _ in names))
        figure.clear()
        plot = figure.add_subplot(211, ylim=(-0.5, 0.5))
        plot.plot(indata)
        im = figure.add_subplot(212)
        im.imshow(mel.data, aspect="auto", origin="lower", cmap="inferno")
        canvas.draw()

    with sd.InputStream(
        samplerate=sr, blocksize=2048, channels=1, callback=callback_wrapper
    ):
        tk.mainloop()
