import random

import cv2
import numpy as np
from PIL import Image


def smooth_edge(img, pixel_smooth):
    # pixel_smooth=300
    im_pil = Image.fromarray(img)
    im_pil = im_pil.resize((pixel_smooth, pixel_smooth), Image.ANTIALIAS)
    im_pil = im_pil.resize((512, 512), Image.ANTIALIAS)
    return np.asarray(im_pil)


def zoom_scale(img, pixel_crop):
    im_pil = Image.fromarray(img)
    im_pil = im_pil.crop((pixel_crop, pixel_crop, 512 - pixel_crop, 512 - pixel_crop))
    im_pil = im_pil.resize((512, 512), Image.ANTIALIAS)
    return np.asarray(im_pil)


def gen_mask(im_origin: np.ndarray):
    rand_mask = random.choice([0, 1])
    if rand_mask:
        logo_s = cv2.imread("asset/mask1.png", cv2.IMREAD_UNCHANGED)
    else:
        logo_s = cv2.imread("asset/mask2.png", cv2.IMREAD_UNCHANGED)

    # RANDOMIZE MASK
    pixel_shift_x = np.random.randint(-3, 3)
    pixel_shift_y = np.random.randint(-3, 3)
    pixel_zoom_scale = np.random.randint(-3, 3)
    pixel_smooth = np.random.randint(300, 400)
    flip_up = np.random.randint(9)
    opacity = np.random.uniform(15, 35) / 100
    # opacity=1.

    center_x = 256
    center_y = 256

    if flip_up % 3 != 1:
        center_x += pixel_shift_x
        for i in range(logo_s.shape[1] - 1, pixel_shift_x, -1):
            logo_s = np.roll(logo_s, -1, axis=1)
            logo_s[:, -1] = logo_s[:, 0]

    if flip_up % 3 != 2:
        center_y += pixel_shift_y
        for i in range(logo_s.shape[1] - 1, pixel_shift_y, -1):
            logo_s = np.roll(logo_s, -1, axis=0)
            logo_s[:, -1] = logo_s[:, 0]

    logo_s = smooth_edge(logo_s, pixel_smooth)
    logo_s = zoom_scale(logo_s, pixel_zoom_scale)
    h, w, _ = im_origin.shape

    result = np.zeros((h, w, 3), np.uint8)
    alpha = logo_s[:, :, 3] / 255.0

    alpha *= opacity
    result[:, :, 0] = (1. - alpha) * im_origin[:, :, 0] + alpha * logo_s[:, :, 0]
    result[:, :, 1] = (1. - alpha) * im_origin[:, :, 1] + alpha * logo_s[:, :, 1]
    result[:, :, 2] = (1. - alpha) * im_origin[:, :, 2] + alpha * logo_s[:, :, 2]

    top = max(center_y - 64, 0)
    left = max(center_x - 64, 0)

    if top + 128 >= 512:
        height = 512 - top
    else:
        height = 128

    if left + 128 >= 512:
        width = 521 - left
    else:
        width = 128

    # bbox = np.array([top, left, height, width])
    # Fucking gix bbox to be center box of image with shape 256 x 256
    bbox = np.array([64, 64, 128, 128])
    return result, alpha, bbox


if __name__ == '__main__':
    src = cv2.imread("asset/image.jpg")

    for _ in range(10):
        res, _, box = gen_mask(src)
        cv2.rectangle(res, (box[1], box[0]), (box[1] + box[3], box[0] + box[2]), (0, 255, 0), 2)
        cv2.imshow("Result", res)
        cv2.waitKey(0)
