# dataset settings

img_scale = (1024, 1024)
crop_size = (512, 512)
img_norm_cfg = dict(
    mean=[95.523, 57.776, 25.422], std=[58.273, 35.645, 16.790], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomRotate',
        prob=1,
        degree=(-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15)),
    dict(
        type='Normalize',**img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

refuge_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomRotate',
        prob=1,
        degree=(-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15)),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # TODO
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_IDRiD_train = dict(
    type='RepeatDataset',
        times=400,
        dataset=dict(
            img_dir='train/images',
            ann_dir='train/ann',
            data_root='data/FOVCrop-padding/RETA',
            type='vesselDataset',
            pipeline=train_pipeline))

dataset_DDR_train = dict(
    type='RepeatDataset',
        times=30,
        dataset=dict(
            img_dir='train/images',
            ann_dir='train/ann',
            data_root='data/FOVCrop-padding/DDR-FOVCrop-padding',
            type='LesionDataset',
            pipeline=train_pipeline))


dataset_REFUGE_train=dict(
    type='RepeatDataset',
        times=30,
        dataset=dict(
            img_dir='train/images',
            ann_dir='train/ann',
            data_root='data/FOVCrop-padding/REFUGE-FOVCrop-padding',
            type='ODOCDataset',
            pipeline=train_pipeline))

dataset_DRIVE_test = dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='data/FOVCrop-padding/DRIVE-FOVCrop-padding',
    type='vesselDataset',
    pipeline=test_pipeline,
    img_suffix='.tif')

dataset_STARE_test = dict(
    img_dir='images',
    ann_dir='ann',
    data_root='data/FOVCrop-padding/STARE-FOVCrop-padding',
    type='vesselDataset',
    pipeline=test_pipeline)

dataset_ddr_test=dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='data/FOVCrop-padding/DDR-FOVCrop-padding',
    type='LesionDataset',
    pipeline=test_pipeline)

dataset_REFUGE_test = dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='data/FOVCrop-padding/REFUGE-FOVCrop-padding',
    type='ODOCDataset',
    pipeline=test_pipeline)

dataset_idrid_test=dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='data/FOVCrop-padding/IDRiD-FOVCrop-padding',
    type='LesionDataset',
    pipeline=test_pipeline)

dataset_aria_test = dict(
    img_dir='images',
    ann_dir='ann',
    data_root='data/FOVCrop-padding/ARIA-FOVCrop-padding',
    type='vesselDataset',
    pipeline=test_pipeline,
    img_suffix='.tif')


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=[dataset_IDRiD_train, dataset_DDR_train, dataset_REFUGE_train],
    val=[
        dataset_DRIVE_test,
        dataset_STARE_test,
        dataset_ddr_test,
        dataset_REFUGE_test
    ],
    test=dataset_idrid_test)
