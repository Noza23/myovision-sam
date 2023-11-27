from dataclasses import dataclass
import os, logging

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceFocalLoss
from monai.metrics import compute_iou

from .build_myosam import build_myosam
from .dataset import MyoData

logging.basicConfig(
    filename="myosam.log",
    filemode="a",
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

@dataclass
class TrainerConfig:
    DATA_DIR: str
    TRAIN_SPLIT: float
    MAX_EPOCHS: int
    SAVE_EVERY: int
    NUM_WORKERS: int
    RUN_NAME: str
    SNAPHOT_PATH: str="./snapshot/myosam_vit_h.pt"


@dataclass
class OptimizerConfig:
    LR: float=1e-4
    WEIGHT_DECAY: float=0.1


class Trainer:
    def __init__(
        self,
        config_train: TrainerConfig,
        config_optim: OptimizerConfig,
    ):
        self.save_every = config_train.SAVE_EVERY
        self.max_epochs = config_train.MAX_EPOCHS
        self.snapshot_path = config_train.SNAPHOT_PATH
        self.logger = logging.getLogger("Training")
        self.writer = SummaryWriter(f"/runs/{config_train.RUN_NAME}")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model, self.metadata = build_myosam(self.snapshot_path)
        self.epochs_run = self.metadata["EPOCHS_RUN"]
        self.model = self.model.to(self.local_rank)
        
        # amp scaler:
        # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
        self.scaler = torch.cuda.amp.GradScaler()

        # Data
        self.dataset_train = MyoData(
            config_train.DATA_DIR, split=config_train.TRAIN_SPLIT
        )
        self.dataset_test = MyoData(
            config_train.DATA_DIR, train=False, split=config_train.TRAIN_SPLIT
        )
        self.sampler_train = DistributedSampler(self.dataset_train)
        self.sampler_test = DistributedSampler(self.dataset_test)
        # Shuffling is handled by the sampler.
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=1,
            pin_memory=True,
            num_workers=config_train.NUM_WORKERS,
            sampler=self.sampler_train
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=1,
            pin_memory=True,
            num_workers=config_train.NUM_WORKERS,
            sampler=self.sampler_test
        )

        # Optimizer, Scheduler, Page 17 of the paper.
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            betas=(0.9, 0.999),
            weight_decay=config_optim.WEIGHT_DECAY,
            lr=config_optim.LR
        )
        if "OPTIMIZER_STATE" in self.metadata.keys():
            self.optimizer.load_state_dict(self.metadata["OPTIMIZER_STATE"])
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.8)
        if "SCHEDULER_STATE" in self.metadata.keys():
            self.scheduler.load_state_dict(self.metadata["SCHEDULER_STATE"])

        self.mask_head_loss = DiceFocalLoss(
            sigmoid=True, squared_pred=False, lambda_dice=1, lambda_focal=20
        )
        self.model = DDP(self.model, device_ids=[self.local_rank])
        
    def train(self):
        """Trains the model for max_epochs."""
        for epoch in range(self.epochs_run, self.max_epochs):
            _ = self._run_epoch(epoch)
            self.scheduler.step()
            if self.local_rank == 0 & self.global_rank == 0:
                self.writer.add_scalar(
                    "LR", self.scheduler.get_last_lr()[0], epoch
                )
                if epoch % self.save_every == 0:
                    self._save_snapshot(epoch)

    def _run_epoch(self, epoch: int) -> None:
        """Runs a single epoch."""
        epoch_loss_train: float = 0.0
        epoch_loss_test: float = 0.0
        self.logger.info(
            (f"[NODE{self.global_rank}]:[GPU{self.local_rank}] | "
             f"Epoch {epoch} | Steps: {len(self.dataloader_train)}")
        )
        self.sampler_train.set_epoch(epoch)
        # Training
        for b_id, (image, gt_instances) in enumerate(self.dataloader_train):
            # 1x3x1024x1024 ; 1xNx1024x1024
            image: torch.Tensor = image.to(self.local_rank)
            gt_instances: torch.Tensor = gt_instances.to(self.local_rank)
            loss = self._run_batch(
                image, gt_instances.permute(1, 0, 2, 3), train=True
            )
            epoch_loss_train += loss
            # Logging losses
            self.logger.info(
                (f"[NODE{self.global_rank}]:[GPU{self.local_rank}] | "
                 f"Epoch {epoch} | Batch {b_id} | Training Loss {loss:.5f}")
            )
        # Testing
        for b_id, (image, gt_instances) in enumerate(self.dataloader_test):
            image: torch.Tensor = image.to(self.local_rank)
            gt_instances: torch.Tensor = gt_instances.to(self.local_rank)
            loss = self._run_batch(
                image, gt_instances.permute(1, 0, 2, 3), train=False
            )
            self.logger.info(
                (f"[NODE{self.global_rank}]:[GPU{self.local_rank}] | "
                 f"Epoch {epoch} | Batch {b_id} | Testing Loss {loss:.5f}")
            )
            epoch_loss_test += loss
        
        # Logging epoch losses to tensorboard
        self.writer.add_scalar(
            f"Loss/Train[NODE{self.global_rank}]:[GPU{self.local_rank}]",
            epoch_loss_train / len(self.dataloader_train),
            epoch
        )
        self.writer.add_scalar(
            f"Loss/Test[NODE{self.global_rank}]:[GPU{self.local_rank}]",
            epoch_loss_test / len(self.dataloader_test),
            epoch
        )
        
    def _run_batch(
        self,
        image: torch.Tensor,
        gt_instances: torch.Tensor,
        train: bool=True
    ) -> float:
        """
        Runs a single batch.
        
        Args:
            image (torch.Tensor): image of shape 1x3x1024x1024.
            gt_instances (torch.Tensor): ground truth instances of
                shape Nx1x1024x1024.
            train (bool, optional): Whether to train or inference.
        
        Returns:
            (float): Average Batch Loss.
        """
        if train:
            self.model.module.train()
        else:
            self.model.module.eval()
        with torch.set_grad_enabled(train), torch.autocast("cuda"):
            mask_logits: torch.Tensor
            iou_preds: torch.Tensor
            mask_logits, iou_preds = self.model(image, gt_instances)
            # (N, 1, 1024, 1024), (N, 1, 1024, 1024)
            assert(len(mask_logits) == len(gt_instances))
            losses = [
                self.mask_head_loss(x.unsqueeze(1), gt_instances)
                for x in mask_logits.unbind(1)
            ]
            # Loss function: Page 17 of the paper.
            # Still needs discussion
            loss_min, loss_i = min((loss, i) for i, loss in enumerate(losses))
            mask_iou = compute_iou(
                mask_logits[:, loss_i].unsqueeze(1),
                gt_instances
            )
            if train:
                # 20 * FocalLoss + 1 * DiceLoss + 1 * MSE Loss of IoU
                loss: torch.Tensor  = (loss_min
                                       + F.mse_loss(iou_preds, mask_iou))
                # Set to None to prevent memory leak.
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        return loss.item()

    def _save_snapshot(self, epoch: int):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict()
        }
        torch.save(snapshot, self.snapshot_path)
        self.logger.info(
            f"Epoch {epoch} | Snapshot saved at {self.snapshot_path}"
        )