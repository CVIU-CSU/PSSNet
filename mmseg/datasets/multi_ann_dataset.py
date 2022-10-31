from collections import OrderedDict

import os
import os.path as osp
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose

@DATASETS.register_module()
class MultiAnnDataset(CustomDataset):

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 CT=False):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.custom_classes = False
        self.CT = CT

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            for i in range(len(self.ann_dir)):
                if not (self.ann_dir[i] is None or osp.isabs(self.ann_dir[i])):
                    self.ann_dir[i] = osp.join(self.data_root, self.ann_dir[i])
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        if self.CT:
            img_infos = []
            for img in os.listdir(self.img_dir):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])
            print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
            return img_infos
        else:
            return super().load_annotations(self.img_dir, self.img_suffix,
                                     self.ann_dir,
                                     self.seg_map_suffix, self.split)


