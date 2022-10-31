# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='DenseNet',
        arch='121',
        out_indices=(0, 1, 2, 3),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='DenseUnetHead',
        arch = '121',
        in_channels=[64, 256, 512, 1024, 1024],
        in_index=[0, 1, 2, 3, 4],
        channels=64,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
