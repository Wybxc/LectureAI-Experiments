import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch.utils.data import TensorDataset
from voice_changer.sound import Sound


class MFCCData:
    """MFCC 语音数据。"""

    mfcc_dict: Dict[str, np.ndarray]
    index_to_key: Dict[int, str]
    key_to_index: Dict[str, int]

    def __init__(
        self,
        mfcc_dict: Dict[str, np.ndarray],
        index_to_key: Optional[Dict[int, str]] = None,
        key_to_index: Optional[Dict[str, int]] = None,
    ):
        self.mfcc_dict = mfcc_dict
        self.index_to_key = index_to_key or dict(enumerate(mfcc_dict.keys()))
        self.key_to_index = key_to_index or {
            key: index for index, key in self.index_to_key.items()
        }

    @classmethod
    def load(cls, file: BinaryIO):
        """从文件中加载。"""
        mfcc_dict, index_to_key, key_to_index = pickle.load(file)
        return cls(mfcc_dict, index_to_key, key_to_index)

    def save(self, file: BinaryIO):
        """保存到到文件中。"""
        pickle.dump((self.mfcc_dict, self.index_to_key, self.key_to_index), file)

    def __len__(self):
        return len(self.mfcc_dict)

    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack(list(self.mfcc_dict.values()))
        y = np.concatenate(
            [
                [self.key_to_index[key]] * self.mfcc_dict[key].shape[0]
                for key in self.mfcc_dict.keys()
            ]
        )
        return X, y

    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = self.to_array()
        return torch.from_numpy(X).float(), torch.from_numpy(y).long()

    def to_dataset(self) -> TensorDataset:
        return TensorDataset(*self.to_tensor())


def prepare(source_dir: str, output_filename: str, verbose: bool = True) -> MFCCData:
    """预处理语音数据。

    语音数据为一系列 wav 文件，每个文件包含一个音节，文件名为音节的拼音。
    """
    save_path = Path(output_filename)
    if save_path.exists():
        with save_path.open("rb") as f:
            return MFCCData.load(f)

    mfcc_dict = defaultdict(list)

    source_dir_path = Path(source_dir)
    sounds = list(source_dir_path.glob("**/*.wav"))

    for sound_filename in tqdm.tqdm(sounds) if verbose else sounds:
        sound = Sound.from_file(sound_filename)
        f0 = sound.yin(pyin=True)
        mfcc = sound.melspectrogram().mfcc()
        categories = re.findall(r"[aoeiuv]+n?g?", Path(sound_filename).stem)
        if categories:
            category = categories[0]
            for i in range(mfcc.shape[1] + 4):
                if f0.data[i] > 0:
                    mfcc_dict[category].append(mfcc.data[:, i:i+5])
                else:
                    mfcc_dict["-"].append(mfcc.data[:, i:i+5])

    mfcc_data = MFCCData({k: np.array(v) for k, v in mfcc_dict.items()})

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        mfcc_data.save(f)

    return mfcc_data
