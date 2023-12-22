import os

import torch
from torch.distributed import init_process_group, destroy_process_group
import hydra
from omegaconf import DictConfig

from myo_sam.trainer import Trainer, TrainerConfig, OptimizerConfig


def ddp_setup():
    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


@hydra.main(config_path=".", config_name="config")
def main(config: DictConfig):
    ddp_setup()

    trainer_config = TrainerConfig(**config["trainer_config"])
    optimizer_config = OptimizerConfig(**config["optimizer_config"])
    trainer = Trainer(trainer_config, optimizer_config)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    main()
