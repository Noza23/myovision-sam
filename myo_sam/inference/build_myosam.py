import torch
from functools import partial
from segment_anything.modeling import (
    ImageEncoderViT,
    PromptEncoder,
    MaskDecoder,
    TwoWayTransformer,
    Sam,
)


def build_myosam_inference(checkpoint: str) -> Sam:
    """
    Builds a Myosam model from a snapshot.
        - Only difference from original build_sam is the
            pixel-normalization constants

    Params:
        checkpoint (str): Path to the checkpoint
    """
    image_size = 1024
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=32,
            embed_dim=1280,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=16,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[7, 15, 23, 31],
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
        pixel_mean=[13.21, 21.91, 15.04],
        pixel_std=[7.26, 16.40, 12.12],
    )
    sam.eval()
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f)
    sam.load_state_dict(state_dict)
    return sam
