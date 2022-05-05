from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .data import MFCCData
from .model import MLPClassifier


def train(
    mfcc_data: MFCCData,
    model: MLPClassifier,
    *,
    val_set_ratio: float = 0.2,
    batch_size: int = 200,
    max_epochs: int = 10,
    root_dir: str = "./lab/classifier",
    ckpt_path: str = "",
) -> None:
    """训练模型。

    Args:
        mfcc_data: MFCC 语音数据。
        model: 模型。
        val_set_ratio: 验证集比例。
        batch_size: 批大小。
        max_epochs: 最大训练轮数。
        root_dir: 模型保存路径。
        ckpt_path: 加载的记录点路径。
    """
    data = mfcc_data.to_dataset()
    train_set, val_set = random_split(
        data,
        [len(data) - int(val_set_ratio * len(data)), int(val_set_ratio * len(data))],
    )
    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_set, batch_size=batch_size)

    Path(root_dir).mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=-1, default_root_dir=root_dir)
    if ckpt_path:
        trainer.fit(model, train_data_loader, val_data_loader)
    else:
        trainer.fit(model, train_data_loader, val_data_loader, ckpt_path=ckpt_path)
