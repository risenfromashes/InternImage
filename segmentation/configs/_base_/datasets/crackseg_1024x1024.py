# CrackSeg configuration adapted from Cityscapes base for 3840×2160 images
# Dataset: CrackSeg with binary masks (crack / no crack)

dataset_type = 'CustomDataset'
data_root = '../../data/CrackSeg/'

# Normalization settings (same as ImageNet pre-training)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# Use 1024x1024 crops for training and testing
crop_size = (1024, 1024)

# Define classes and color palette for binary segmentation
classes = ['no_crack', 'crack']
palette = [[0, 0, 0], [255, 255, 255]]

# Training data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Resize long edge to dataset resolution then random scale
    dict(type='Resize', img_scale=(3840, 2160), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# Testing / Validation pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3840, 2160),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/img',
        ann_dir='train/mask',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='valid/img',
        ann_dir='valid/mask',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='valid/img',
        ann_dir='valid/mask',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette
    )
)

