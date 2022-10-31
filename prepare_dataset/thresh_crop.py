import argparse
import os
import numpy as np

import cv2
import PIL.Image
import imgviz


def get_bound(image, threshold=20, mode_keep=False, filename=''):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.dilate(thresh, kernel)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = [0, 0, 0, 0]
    for cnt in contours:
        res = cv2.boundingRect(cnt)
        if res[2] > result[2] or res[3] > result[3]:
            result = res
    x, y, w, h = result

    if mode_keep:
        if np.abs(image.shape[0] / image.shape[1] - 1.0) < 0.1:
            x, y, w, h = 0, 0, image.shape[1], image.shape[0]

    return x, y, w, h


def crop(image_path, label_path=None,
         output_image=None, output_label=None, output_log_file=None, thresh=20, mode_keep=False,
         image_suffix='.jpg', label_suffix='.png', color_mode=False):
    os.makedirs(output_image, exist_ok=True)
    if output_label:
        os.makedirs(output_label, exist_ok=True)

    lines = []
    for root, dirs, files in os.walk(image_path):
        for i, image_file in enumerate(files):
            image = cv2.imread(os.path.join(image_path, image_file))

            label = None
            label_file = None
            if label_path:
                label_file = image_file.replace(image_suffix, label_suffix)
                # if not os.path.exists(label_file):
                #     continue
                label = PIL.Image.open(os.path.join(label_path, label_file))  # 8 bit image
                label = np.array(label)

            ori_h, ori_w = image.shape[0], image.shape[1]
            x, y, w, h = get_bound(image, thresh, mode_keep, image_file)

            h0 = y
            h1 = y + h
            w0 = x
            w1 = x + w

            line = '{} {} {} {} {} {} {}\n'.format(
                image_file.replace(image_suffix, ''), h0, ori_h - h1, w0, ori_w - w1, ori_h, ori_w)
            lines.append(line)
            print(line, end='')

            image = image[h0:h1, w0:w1]
            cv2.imwrite(os.path.join(output_image, image_file), image)

            if label is not None:
                label = label[h0:h1, w0:w1]
                label = PIL.Image.fromarray(label, mode='P')
                if color_mode:
                    colormap = imgviz.label_colormap()
                    label.putpalette(colormap)
                label.save(os.path.join(output_label, label_file))

    with open(output_log_file, mode='w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='D:/dataset/ddr/DDR',
                        help='path of input root')
    parser.add_argument('--output', type=str, default='D:/dataset/ddr/DDR',
                        help='path of output root')
    args = parser.parse_args()

    # train
    crop(image_path=os.path.join(args.input, 'image/train'),
         label_path=os.path.join(args.input, 'annotation/train/label'),
         output_image=os.path.join(args.output, 'image/train_crop'),
         output_label=os.path.join(args.output, 'annotation/train/label_crop'),
         output_log_file=os.path.join(args.output, 'train_crop.txt'),
         thresh=20)
    '''
    # val
    crop(image_path=os.path.join(args.input, 'image/val'),
         output_image=os.path.join(args.output, 'image/val_crop'),
         output_log_file=os.path.join(args.output, 'val_crop.txt'),
         thresh=30, mode_keep=True)
    '''
    # test
    crop(image_path=os.path.join(args.input, 'image/test'),
         label_path=os.path.join(args.input, 'annotation/test/label'),
         output_image=os.path.join(args.output, 'image/test_crop'),
         output_label=os.path.join(args.output, 'annotation/test/label_crop'),
         output_log_file=os.path.join(args.output, 'test_crop.txt'),
         thresh=20)
