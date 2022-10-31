# define GAN model
num_groups = 3
norm_cfg = dict(type='GN', num_groups=num_groups, requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
decode_head=dict(
    type='UperHeadGroup',
    in_channels=[96, 192, 384, 768],
    in_index=[0, 1, 2, 3],
    pool_scales=(1, 2, 3, 6),
    channels=128,
    dropout_ratio=0.1,
    num_classes=1,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=[
        dict(type='BinaryLoss', loss_type='dice', smooth=1e-5, loss_weight=1.0),
        dict(type='BinaryLoss', loss_type='dice', smooth=1e-5, loss_weight=1.0),
        dict(type='BinaryLoss', loss_type='dice', smooth=1e-5, ODOC=True, loss_weight=1.0)
    ],
    num_classes_multi=[4, 2],
    groups=num_groups)
auxiliary_head=dict(
    type='FCNHeadGroup',
    in_channels=384,
    in_index=2,
    channels=128,
    num_convs=1,
    concat_input=False,
    dropout_ratio=0.1,
    num_classes=1,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=[
        dict(type='BinaryLoss', loss_type='dice', smooth=1e-5, loss_weight=0.4),
        dict(type='BinaryLoss', loss_type='dice', smooth=1e-5, loss_weight=0.4),
        dict(type='BinaryLoss', loss_type='dice', smooth=1e-5, ODOC=True, loss_weight=0.4)
    ],
    num_classes_multi=[4, 2],
    groups=num_groups)
discriminator_1 = dict(
    type='Discriminator',
    in_channels=1,  # Need to be set.
)
discriminator_2 = dict(
    type='Discriminator',
    in_channels=4,  # Need to be set.
)
discriminator_3 = dict(
    type='Discriminator',
    in_channels=2,  # Need to be set.
)
auxiliary_discriminator_1 = dict(
    type='Discriminator',
    in_channels=1
)
auxiliary_discriminator_2 = dict(
    type='Discriminator',
    in_channels=4
)
auxiliary_discriminator_3 = dict(
    type='Discriminator',
    in_channels=2
)
model = dict(
    type='MultiTaskGAN_3',
    generator=dict(
        type='MultiTaskGenerator',
        pretrained='pretrain/swin_tiny_patch4_window7_224.pth',
        backbone=dict(
            type='SwinTransformer',
            pretrain_img_size=224,
            embed_dims=96,
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            norm_cfg=backbone_norm_cfg),
        decode_head=decode_head,
        auxiliary_head=auxiliary_head,
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    ),
    discriminator=[
        discriminator_1,
        discriminator_2,
        discriminator_3
    ],
    auxiliary_discriminator=[
        auxiliary_discriminator_1,
        auxiliary_discriminator_2,
        auxiliary_discriminator_3
    ],
    gan_loss=dict(type='GANLoss'),
    gen_auxiliary_loss=dict(
        type='MultiGenAuxLoss',
        loss_weight=1.0,
        loss_decode=[
             dict(
                 type='BinaryLoss',
                 loss_type='dice',
                 smooth=1e-5,
                 loss_weight=1.0),
             dict(
                 type='BinaryLoss',
                 loss_type='dice',
                 smooth=1e-5,
                 loss_weight=1.0),
             dict(
                 type='BinaryLoss',
                 loss_type='dice',
                 smooth=1e-5,
                 loss_weight=1.0,
                 ODOC=True),
                 ]))

train_cfg = None
test_cfg = None
