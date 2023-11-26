
import torch
from torch import nn
import torch.nn.functional as F
from segment_anything.modeling import (
    ImageEncoderViT,
    PromptEncoder,
    MaskDecoder
)
from .utils import (
    sample_initial_points,
    sample_points_from_error_region
)

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
        self.threshold: float = 0

    def forward(
        self,
        image: torch.Tensor,
        gt_instances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided image and prompts.
        It uses automative interactive segmentation technique.

        Args:
            image (torch.Tensor): Input image tensor. 1x3xHxW
            gt_instances (torch.Tensor): Ground truth masks tensor.
                Nx1xHxW
                
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predicted masks tensor
                Nx1x1024x1024 and iou_preds Nx1.
        """
        features = self.image_encoder(image)
        mask_logits, iou_preds = self._run_interactive_prompting(
            features=features,
            gt_instances=gt_instances,
            its=8
        )
        return mask_logits, iou_preds
        


    def _run_interactive_prompting(
        self,
        features: torch.Tensor,
        gt_instances: torch.Tensor,
        its: int
    ):
        """
        Runs interactive promptng algorithm.

        Args:
            features (torch.Tensor): Batched features 1x256x64x64.
            gt_instances (torch.Tensor): Batched gt instances Nx1xHxW.
            its (int): Number of iterations to run interactive prompting.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: mask logits Nx1x1024x1024
                and iou_preds Nx1.

        """
        # When multiple masks are returned, the mask passed to the next
        # iteration and used to sample the next point is the one with
        # the highest predicted IoU.
        N, _, _, _ = gt_instances.shape
        # Sample initial points and labels tuple[(N, 1, 2), (N, 1)]
        initial_points = sample_initial_points(gt_instances)
        low_res_masks, iou_preds = self._predict(
            features, initial_points, None
        )
        only_mask_steps = torch.cat(
            [torch.randint(1, its, (1, )), torch.Tensor([its+1])]
        )
        # 8 it mask+point  prompt and 2 it point only.
        for i in range(its+2):
            # The mask with highest predicted IoU is used to sample.
            idx = (low_res_masks.shape[1] * torch.arange(0, N)
                   + torch.argmax(iou_preds, dim=1))
            # (N, 1, 256, 256)
            low_res_masks = low_res_masks.flatten(0, 1)[idx].unsqueeze(1)
            binary_mask = self.postprocess_masks(low_res_masks, binarize=True)
            points = sample_points_from_error_region(
                gt_masks=gt_instances,
                pred_masks=binary_mask
            )
            low_res_masks, iou_preds = self._predict(
                features=features,
                point_prompts=points if i != only_mask_steps else None,
                mask_prompts=low_res_masks
            )
        # upscale
        mask_logits = self.postprocess_masks(low_res_masks)
        return mask_logits, iou_preds


    def _predict(
        self,
        features: torch.Tensor,
        point_prompts: tuple[torch.Tensor, torch.Tensor],
        mask_prompts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts low_res_masks and iou_preds given features and prompts.

        Args:
            features (torch.Tensor): single image features 1x256x64x64.
            point_prompts (tuple[torch.Tensor, torch.Tensor]): tuple of
                initial points and labels for each instance
                (B, 1, 2) and (B, 1) respectively.
            mask_prompts (torch.Tensor): low_res_masks for each
                instance Nx1x256x256.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: low_res_masks NxCx256x256
                and iou_preds NxCx1 where C is the number of predicted
                masks for each instance.
        """
        # Nx2x256, Nx256x64x64
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point_prompts, boxes=None, masks=mask_prompts
        )
        # Features will be expanded to Nx256x64x64 see source code.
        low_res_masks, iou_preds = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )
    
        return low_res_masks, iou_preds
    
    def postprocess_masks(
        self,
        low_res_masks: torch.Tensor,
        binarize: bool=False
    ) -> torch.Tensor:
        """Upscales and binarizes low_res_logit masks."""
        # (N, 1, 256, 256) -> (N, 1, 1024, 1024)
        mask = F.interpolate(
            low_res_masks,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False
        )
        if binarize:
            mask = (mask > self.threshold).to(torch.uint8)
        return mask