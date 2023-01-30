# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
crop_size = (512, 512)

norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained=None,
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=384,
        num_layers=12,
        num_heads=6,
        mlp_ratio=4,
        out_indices=(8, 9, 10, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic'),
    decode_head=dict(
        type="ViTLinearHead",
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        input_transform="resize_concat",
        channels=1536,
        dropout_ratio=0,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(341, 341)),
    # init_cfg=dict(type="Pretrained", checkpoint="/home/thomas/Downloads/temp/converted.pth"),
)

optimizer = dict(
    _delete_=True, type="AdamW", lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=4)
