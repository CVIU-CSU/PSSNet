import numpy as np
from PIL import Image, ImageFilter
import random
from torchvision import transforms

from ..builder import PIPELINES


@PIPELINES.register_module()
class Blur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, results):
        img = results['img']
        img = Image.fromarray(img.astype(np.uint8))
        if random.random() < self.p:
            sigma = np.random.uniform(0.1, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        img = np.array(img)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(p={self.p})'
        return repr_str


@PIPELINES.register_module()
class Cutout(object):
    def __init__(self, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
                 ratio_2=1 / 0.3, value_min=0, value_max=255, pixel_level=True):
        self.p = p
        self.size_min = size_min
        self.size_max = size_max
        self.ratio_1 = ratio_1
        self.ratio_2 = ratio_2
        self.value_min = value_min
        self.value_max = value_max
        self.pixel_level = pixel_level

    def __call__(self, results):
        if random.random() < self.p:
            img = results['img']
            img_h, img_w, img_c = img.shape

            while True:
                size = np.random.uniform(self.size_min, self.size_max) * img_h * img_w
                ratio = np.random.uniform(self.ratio_1, self.ratio_2)
                erase_w = int(np.sqrt(size / ratio))
                erase_h = int(np.sqrt(size * ratio))
                x = np.random.randint(0, img_w)
                y = np.random.randint(0, img_h)

                if x + erase_w <= img_w and y + erase_h <= img_h:
                    break

            if self.pixel_level:
                value = np.random.uniform(self.value_min, self.value_max, (erase_h, erase_w, img_c))
            else:
                value = np.random.uniform(self.value_min, self.value_max)

            img[y:y + erase_h, x:x + erase_w] = value

            for key in results.get('seg_fields', []):
                gt_seg = results[key]
                gt_seg[y:y + erase_h, x:x + erase_w] = 255
                results[key] = gt_seg

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(p={self.p})'
        return repr_str


@PIPELINES.register_module()
class ColorJitter(object):
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, results):
        if random.random() < self.p:
            img = results['img']
            img = Image.fromarray(img.astype(np.uint8))
            img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = np.array(img).astype(np.float32)
            results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(p={self.p})'
        return repr_str


@PIPELINES.register_module()
class RandomGrayscale(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, results):
        img = results['img']
        img = Image.fromarray(img.astype(np.uint8))
        img = transforms.RandomGrayscale(p=self.p)(img)
        img = np.array(img).astype(np.float32)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(p={self.p})'
        return repr_str
