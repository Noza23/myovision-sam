from .myosam import MyoSam
from typing import Union, Any
import torch
from functools import partial
from segment_anything.modeling import (
    ImageEncoderViT,
    PromptEncoder,
    MaskDecoder,
    TwoWayTransformer,
)


def build_myosam(
    snapshot_path: Union[str, None],
) -> tuple[MyoSam, dict[str, Any]]:
    """
    Builds a MyoSam model from a snapshot. The snapshot is a dictionary
    containing the model state and the number of epochs run.

    Args:
        snapshot_path (str): Path to the snapshot.

    Returns:
        (tuple[MyoSam, int]): Tuple of the model and other metadata:
            epochs_run (int): Number of epochs run.
            OPTIMIZER_STATE (dict): Optimizer state.
            SCHEDULER_STATE (dict): Scheduler state.
    """
    encoder_embed_dim = 1280
    encoder_depth = 32
    encoder_num_heads = 16
    encoder_global_attn_indexes = [7, 15, 23, 31]
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # Build the model
    myosam = MyoSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    )
    if snapshot_path is None:
        return myosam, {"EPOCHS_RUN": 0}
    snapshot: dict = torch.load(snapshot_path, map_location="cpu")
    myosam.load_state_dict(snapshot["MODEL_STATE"])
    return myosam, {k: v for k, v in snapshot.items() if k != "MODEL_STATE"}
