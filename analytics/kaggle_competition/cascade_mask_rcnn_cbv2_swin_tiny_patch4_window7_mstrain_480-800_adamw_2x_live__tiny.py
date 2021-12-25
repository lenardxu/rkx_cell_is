_base_ = [
    '../swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
]

# TODO Multi-scale should be adopted! -> shortest side set to one of (440, 480, 520, 580, 620) pixels
# TODO the predefined anchor box sizes were reduced to (4, 9, 17, 31, 64, 127 pixels)
# TODO the anchor generator aspect ratios are changed to (0.25, 0.5, 1, 2, 4)
model = dict(
    backbone=dict(
        type='CBSwinTransformer',
        frozen_stages=4,
    ),
    neck=dict(
        type='CBFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1],
            ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    test_cfg = dict(
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms'),
        )
    ),
    train_cfg = dict(
        rpn_proposal=dict(
            nms_post=1000,
            max_per_img=1000
        )
    ),
    roi_head = dict(
        mask_head=dict(
            num_classes=1
        ),
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]
   )
)
# TODO normalization parameters should be changed!
img_norm_cfg = dict(
    mean=[128.0, 128.0, 128.0], std=[11.58, 11.58, 11.58], to_rgb=True)

# augmentation strategy originates from HTC
# (440, 480, 520, 580, 620)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize',
         img_scale=[(440,600),(480,650),(520,704),(580,800)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(520,704),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

samples_per_gpu=1
data = dict(samples_per_gpu=samples_per_gpu,
            workers_per_gpu=1,
            train=dict(
                type='CocoDataset',
                pipeline=train_pipeline,
                classes=('cell',),
                ann_file='../data/livecell_images/livecell_coco_train_tiny.json',
                img_prefix='../data/livecell_images/livecell_images/'
                       ),
            test=dict(
                type='CocoDataset',
                pipeline=test_pipeline,
                classes=('cell',),
                ann_file='../data/livecell_images/livecell_coco_test_tiny.json',
                img_prefix='../data/livecell_images/livecell_images/'
                ),
            val=dict(
                type='CocoDataset',
                pipeline=test_pipeline,
                classes=('cell',),
                ann_file='../data/livecell_images/livecell_coco_val_tiny.json',
                img_prefix='../data/livecell_images/livecell_images/'))


dataset_type = 'CocoDataset'
data_root = '../data/livecell_images'

runner = dict(max_epochs=24)
lr_config = dict(step=[16,22])
optimizer = dict(lr=0.00005)
load_from = './checkpoints/cascade_mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'

seed=0
gpu_ids = range(1)
work_dir = './live_version1'




