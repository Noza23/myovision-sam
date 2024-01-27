from dataclasses import dataclass
import os
import logging

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
from .utils import sample_initial_points, sample_points_from_error_region


@dataclass
class TrainerConfig:
    DATA_DIR: str
    MAX_INSTANCES: int
    TRAIN_SPLIT: float
    MAX_EPOCHS: int
    SAVE_EVERY: int
    INT_ITERATIONS: int
    NUM_WORKERS: int
    RUN_NAME: str
    SNAPHOT_PATH: str = "./snapshot/myosam_vit_h.pt"
    MIXED_PRECISION: bool = True


@dataclass
class OptimizerConfig:
    LR: float = 1e-4
    WEIGHT_DECAY: float = 0.1


class Trainer:
    def __init__(
        self,
        config_train: TrainerConfig,
        config_optim: OptimizerConfig,
    ) -> None:
        """
        Args:
            config_train (TrainerConfig): Training cfg (config.yaml)
            config_optim (OptimizerConfig): Optimizer cfg (config.yaml)
        """
        self.save_every = config_train.SAVE_EVERY
        self.max_epochs = config_train.MAX_EPOCHS
        self.snapshot_path = config_train.SNAPHOT_PATH
        self.its = config_train.INT_ITERATIONS
        self.writer = SummaryWriter(f"runs/{config_train.RUN_NAME}")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.model, self.metadata = build_myosam(self.snapshot_path)
        self.epochs_run = self.metadata["EPOCHS_RUN"]
        self.model = self.model.to(self.local_rank)
        self.mixed_prec = config_train.MIXED_PRECISION

        # Logger
        file_handler = logging.FileHandler("myosam.log")
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
            )
        )
        file_handler.setLevel(logging.INFO)
        self.logger = logging.getLogger("Training")
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

        # amp scaler:
        # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
        self.scaler = torch.cuda.amp.GradScaler()

        # Data
        self.dataset_train = MyoData(
            config_train.DATA_DIR,
            config_train.MAX_INSTANCES,
            split=config_train.TRAIN_SPLIT,
        )
        self.dataset_test = MyoData(
            config_train.DATA_DIR,
            config_train.MAX_INSTANCES,
            train=False,
            split=config_train.TRAIN_SPLIT,
        )
        self.sampler_train = DistributedSampler(self.dataset_train)
        self.sampler_test = DistributedSampler(self.dataset_test)
        # Shuffling is handled by the sampler.
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=1,
            pin_memory=True,
            num_workers=config_train.NUM_WORKERS,
            sampler=self.sampler_train,
            shuffle=False,
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=1,
            pin_memory=True,
            num_workers=config_train.NUM_WORKERS,
            sampler=self.sampler_test,
            shuffle=False,
        )

        # Optimizer, Scheduler, Page 17 of the paper.
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            betas=(0.9, 0.999),
            weight_decay=config_optim.WEIGHT_DECAY,
            lr=config_optim.LR,
        )
        if "OPTIMIZER_STATE" in self.metadata.keys():
            self.optimizer.load_state_dict(self.metadata["OPTIMIZER_STATE"])
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.9)
        if "SCHEDULER_STATE" in self.metadata.keys():
            self.scheduler.load_state_dict(self.metadata["SCHEDULER_STATE"])

        self.mask_head_loss = DiceFocalLoss(
            sigmoid=True, squared_pred=False, lambda_dice=1, lambda_focal=20
        )
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            find_unused_parameters=True,
        )

    def train(self):
        """Trains the model for max_epochs."""
        for epoch in range(self.epochs_run + 1, self.max_epochs):
            _ = self._run_epoch(epoch)
            # Scheduler step is updated after each epoch.
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            self.logger.info(f"Epoch {epoch} | LR: {lr}")
            if self.local_rank == 0:
                # Save snapshot and log LR only from master GPU.
                self.writer.add_scalar("LR", lr, epoch)
                if epoch % self.save_every == 0:
                    self._save_snapshot(epoch)

    def _run_epoch(self, epoch: int) -> None:
        """Runs a single epoch."""
        epoch_loss_train: float = 0.0
        epoch_loss_test: float = 0.0
        train_n = len(self.dataloader_train)
        test_n = len(self.dataloader_test)
        self.logger.info(
            f"[GPU{self.local_rank}] | Epoch {epoch} | Steps: {train_n}"
        )
        self.sampler_train.set_epoch(epoch)
        # Training
        self.model.train()
        for b_id, (image, gt_instances) in enumerate(self.dataloader_train):
            # 1x3x1024x1024 ; 1xNx1024x1024
            image = image.to(self.local_rank)
            gt_instances = gt_instances.to(self.local_rank)
            loss = self._run_batch(
                image, gt_instances.permute(1, 0, 2, 3), train=True
            )
            epoch_loss_train += loss
            # Logging losses
            self.logger.info(
                (
                    f"[GPU{self.local_rank}] | Epoch {epoch} | Batch {b_id} |"
                    f" Training Loss {loss:.5f}"
                )
            )

        # Testing
        self.model.eval()
        for b_id, (image, gt_instances) in enumerate(self.dataloader_test):
            image = image.to(self.local_rank)
            gt_instances = gt_instances.to(self.local_rank)
            loss = self._run_batch(
                image, gt_instances.permute(1, 0, 2, 3), train=False
            )
            self.logger.info(
                (
                    f"[GPU{self.local_rank}] | Epoch {epoch} | Batch {b_id} |"
                    f" Testing Loss {loss:.5f}"
                )
            )
            epoch_loss_test += loss

        # Logging epoch losses to tensorboard
        self.writer.add_scalar(
            f"avg. Loss/Train[GPU{self.local_rank}]",
            epoch_loss_train / train_n,
            epoch,
        )
        self.writer.add_scalar(
            f"avg. Loss/Test[GPU{self.local_rank}]",
            epoch_loss_test / test_n,
            epoch,
        )

    def _run_batch(
        self, image: torch.Tensor, gt_instances: torch.Tensor, train: bool
    ) -> float:
        """
        Runs a single batch.
        Args:
            image (torch.Tensor): image of shape 1x3x1024x1024.
            gt_instances (torch.Tensor): ground truth instances of
                shape Nx1x1024x1024.
            train (bool): Whether to train or inference.

        Returns:
            (float): Average Batch Loss.
        """
        average_loss: float = 0.0
        # initial step + its + last step
        accumulation_steps = self.its + 3
        # sample a step inbetween the algo to only prompt with mask
        only_mask_step = torch.randint(1, self.its, (1,)).item()
        low_res_masks = torch.zeros(
            (gt_instances.shape[0], 1, 256, 256), device=self.local_rank
        )
        with torch.set_grad_enabled(train), torch.autocast("cuda"):
            # Gradient accumulation
            with self.model.no_sync():
                # Interactive prompting
                for i in range(self.its + 2):
                    if i == 0:
                        # Initial step
                        points = sample_initial_points(gt_instances.detach())
                    else:
                        # Interactive step
                        points = sample_points_from_error_region(
                            gt_instances.detach(),
                            self.model.module.upscale(low_res_masks.detach()),
                        )
                    low_res_masks, iou_pred = self.model(
                        image,
                        points=points if i != only_mask_step else None,
                        masks=low_res_masks.detach() if i != 0 else None,
                    )
                    gt_iou = compute_iou(
                        self.model.module.upscale(low_res_masks.detach()),
                        gt_instances,
                    )
                    # Loss is averaged over accumulation steps.
                    loss = (
                        self.compute_loss(
                            self.model.module.upscale(low_res_masks, False),
                            gt_instances,
                            iou_pred,
                            gt_iou,
                        )
                        / accumulation_steps
                    )
                    average_loss += loss.item()
                    if train:
                        if self.mixed_prec:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

            # Last step is out of the no_sync context to sync accumlated grads.
            points = sample_points_from_error_region(
                gt_instances.detach(),
                self.model.module.upscale(low_res_masks.detach()),
            )
            low_res_masks, iou_pred = self.model(
                image, points=None, masks=low_res_masks.detach()
            )
            gt_iou = compute_iou(
                self.model.module.upscale(low_res_masks.detach()), gt_instances
            )
            loss = (
                self.compute_loss(
                    self.model.module.upscale(low_res_masks, False),
                    gt_instances,
                    iou_pred,
                    gt_iou,
                )
                / accumulation_steps
            )
            if train:
                if self.mixed_prec:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    # Update is called after accumulating over acc_steps.
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
        return average_loss

    def _save_snapshot(self, epoch: int) -> None:
        """Saves a snapshot of the model."""
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
        }
        torch.save(snapshot, self.snapshot_path)
        self.logger.info(
            f"Epoch {epoch} | Snapshot saved at {self.snapshot_path}"
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        gt_instances: torch.Tensor,
        iou_pred: torch.Tensor,
        gt_iou: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes loss: DiceLoss + 20xFocalLoss + MSE(IoU)
        Args:
            logits (torch.Tensor): Upscaled logits Nx1x1024x1024.
            gt_instances (torch.Tensor): Ground truth instances
                Nx1x1024x1024.
            iou_pred (torch.Tensor): Predicted IoU scores Nx1x1.
            gt_iou (torch.Tensor): Ground truth IoU scores Nx1x1.
        """
        loss = self.mask_head_loss(logits, gt_instances) + F.mse_loss(
            iou_pred, gt_iou
        )
        return loss
