import os
from typing import BinaryIO
import streamlit as st
from voice_changer.sound import Sound
from voice_changer.filter import MonoFilter, PitchChanger
import tempfile
import matplotlib.pyplot as plt

st.set_page_config(
    layout="wide",
    menu_items={        
        "About": "人工智能引论课程作业。",
    },
)


def load_sound(file: BinaryIO):
    temp = tempfile.NamedTemporaryFile(mode="w+b", suffix=".wav", delete=False)
    temp.close()
    try:
        with open(temp.name, "wb") as f:
            f.write(file.read())
        return Sound.from_file(temp.name)
    finally:
        os.remove(temp.name)


def export_sound(sound: Sound) -> bytes:
    temp = tempfile.NamedTemporaryFile(mode="w+b", suffix=".wav", delete=False)
    temp.close()
    try:
        sound.save(temp.name, format="wav")
        with open(temp.name, "rb") as f:
            return f.read()
    finally:
        os.remove(temp.name)


def display_mel(sound: Sound) -> None:
    with st.expander("查看语谱图"):
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(
            sound.melspectrogram().data, aspect="auto", origin="lower", cmap="inferno"
        )
        sound.set_axis_by_frames(ax)
        st.pyplot(fig)


left, right = st.columns(2)

sound_src = sound_dst = None

with left:
    file_src = st.file_uploader("选择源音频", type=["mp3", "wav", "flac", "ogg", "aac"])
    if file_src:
        sound_src = load_sound(file_src)
        st.audio(export_sound(sound_src), format="audio/wav")
        display_mel(sound_src)

        file_dst = st.file_uploader("选择目标音频", type=["mp3", "wav", "flac", "ogg", "aac"])
        if file_dst:
            sound_dst = load_sound(file_dst)
            st.audio(export_sound(sound_dst), format="audio/wav")
            display_mel(sound_dst)
        else:
            sound_dst = None
    else:
        sound_src = None

with right:
    if sound_src and sound_dst:
        with st.spinner("正在处理中..."):
            pitch_changer = PitchChanger().fit(sound_src, sound_dst)
            sound_mid = pitch_changer.predict(sound_src)

            mono_filter = MonoFilter().fit(sound_src, sound_dst)
            sound_tgt = mono_filter.predict(sound_mid)

        st.markdown("## 变声结果：")
        st.audio(export_sound(sound_tgt), format="audio/wav")
        display_mel(sound_tgt)
