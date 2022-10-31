_base_ = [
    '../_base_/datasets/multi_task_vessel_lesion_OD_idrid_ddr_refuge_semi_OD.py',
    '../_base_/models/multi_task_gan_bdice_vessel_lesion_OD_group.py',
    '../_base_/default_runtime_gan.py'
]

train_cfg = dict(_delete_=True, lamda=[0.1, 0.1, 0.1], lamda_aux=[0.02, 0.02, 0.02], semi=True, semi_weight=[0.05, 0.05, 0.05])

model = dict(
    generator=dict(
        decode_head=dict(window_size=16),
    ))

# define optimizer
optimizer = dict(
    generator=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys={
                'absolute_pos_embed': dict(decay_mult=0.),
                'relative_position_bias_table': dict(decay_mult=0.),
                'norm': dict(decay_mult=0.)
            })),
    discriminator=dict(
        type='Adam', lr=0.0002, betas=(0.9, 0.99)),
    auxiliary_discriminator=dict(
        type='Adam', lr=0.0002, betas=(0.9, 0.99))
)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

evaluation = dict(interval=6000, metric='mIoU', by_epoch=False)

checkpoint_config = dict(interval=6000, by_epoch=False)

total_iters = 60000

# load_generator = 'work_dirs/upernet_swin_tiny_patch4_window7_512x512_40k_multi_pretrain_224x224_1K_bdice_vessel_lesion_idrid_ddr/iter_40000.pth'
load_generator = None