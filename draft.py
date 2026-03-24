from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. from src import utils)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set root_dir to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

import matplotlib.pyplot as plt
import random


def train(cfg: DictConfig):
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    datamodule.prepare_data()
    datamodule.setup()

    return datamodule.data_test


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)

    # train the model
    data_test = train(cfg)

    print(len(data_test))

    num_images = 9
    plt.figure(figsize=(10, 10))
    
    # Lấy 9 chỉ số ngẫu nhiên
    indices = random.sample(range(len(data_test)), num_images)
    
    for i, idx in enumerate(indices):
        # Lấy sample (thường sample trả về là (image, target) hoặc dict)
        sample = data_test[idx]
        
        # Tùy vào Dataset của bạn, nếu là Tuple thì lấy img = sample[0]
        # Nếu là Dict thì lấy img = sample["image"]
        if isinstance(sample, (tuple, list)):
            img = sample[0]
        elif isinstance(sample, dict):
            img = sample.get("image") or sample.get("img")
        else:
            img = sample

        # Nếu img là Tensor (CHW), cần chuyển về (HWC) để matplotlib hiểu
        if torch.is_tensor(img):
            # Denormalize nếu cần (ở đây vẽ thô nên chỉ cần chuyển định dạng)
            img = img.permute(1, 2, 0).numpy()
            
            # Nếu ảnh bị Normalize (có giá trị âm hoặc > 1), đưa về [0, 1] để vẽ
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)

        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Index: {idx}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()