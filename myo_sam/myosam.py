import torch
from torch import nn
import torch.nn.functional as F
from segment_anything.modeling import (
    ImageEncoderViT,
    PromptEncoder,
    MaskDecoder,
)
from typing import Union


class MyoSam(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
    ) -> None:
        """
        MyoSam is slightly modified version of original Sam class,
            that is used for training the model.
        Args:
            image_encoder (ImageEncoderViT): Backbone used to encode
                the image into embeddings that allow for efficient
                mask prediction.
            prompt_encoder (PromptEncoder): Encodes various types of
                input prompts.
            mask_decoder (MaskDecoder): Predicts masks from the image
                embeddings and encoded prompts.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.input_size = self.image_encoder.img_size
        self.threshold: float = 0.0

    def forward(
        self,
        image: torch.Tensor,
        points: Union[tuple[torch.Tensor, torch.Tensor], None],
        masks: Union[torch.Tensor, None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks end-to-end from provided image and prompts.
        Args:
            image (torch.Tensor): Input image tensor 1x3x1024x1024.
            points (tuple[torch.Tensor, torch.Tensor]): Tuple containing
            point coordinates and point labels. (N, 1, 2) and (N, 1).
            masks (torch.Tensor): Ground truth masks Nx1xHxW.

        Returns:
            low_res_masks (torch.Tensor): Predicted masks Nx1x256x256.
            iou_preds (torch.Tensor): Predicted IoU scores Nx1x1.
        """
        features = self.image_encoder(image)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points, boxes=None, masks=masks
        )

        low_res_masks, iou_preds = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks, iou_preds

    def upscale(
        self, low_res_masks: torch.Tensor, binarize: bool = True
    ) -> torch.Tensor:
        """Upscales and binarizes low_res_logit masks."""
        # (N, 1, 256, 256) -> (N, 1, 1024, 1024)
        mask = F.interpolate(
            low_res_masks,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        if binarize:
            mask = (mask > self.threshold).to(torch.uint8)
        return mask
