import cv2
import numpy as np
from PIL import Image

logo_s_list = [cv2.imread("asset/mask1.png", cv2.IMREAD_UNCHANGED),
               cv2.imread("asset/mask2.png", cv2.IMREAD_UNCHANGED),
               cv2.imread("asset/mask3.png", cv2.IMREAD_UNCHANGED)]


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


def add_logo(im_origin, alpha_0):
    rand_mask = np.random.randint(3)

    logo_s = logo_s_list[rand_mask]
    # RANDOMIZE MASK
    pixel_shift_x = np.random.randint(-15, 15)
    pixel_shift_y = np.random.randint(-30, 30)
    pixel_zoom_scale = np.random.randint(-3, 3)
    pixel_smooth = np.random.randint(200, 400)
    flip_up = np.random.randint(9)
    opacity = np.random.uniform(5, 25) / 100
    # opacity=1.

    if flip_up % 3 != 1:
        for i in range(logo_s.shape[1] - 1, pixel_shift_x, -1):
            logo_s = np.roll(logo_s, -1, axis=1)
            logo_s[:, -1] = logo_s[:, 0]

    if flip_up % 3 != 2:
        for i in range(logo_s.shape[1] - 1, pixel_shift_y, -1):
            logo_s = np.roll(logo_s, -1, axis=0)
            logo_s[:, -1] = logo_s[:, 0]

    logo_s = smooth_edge(logo_s, pixel_smooth)
    logo_s = zoom_scale(logo_s, pixel_zoom_scale)
    h, w, _ = im_origin.shape

    result = np.zeros((h, w, 3), np.uint8)
    alpha = logo_s[:, :, 3] / 255.0

    alpha *= opacity
    alpha_0 += alpha
    result[:, :, 0] = (1. - alpha) * im_origin[:, :, 0] + alpha * logo_s[:, :, 0]
    result[:, :, 1] = (1. - alpha) * im_origin[:, :, 1] + alpha * logo_s[:, :, 1]
    result[:, :, 2] = (1. - alpha) * im_origin[:, :, 2] + alpha * logo_s[:, :, 2]

    return result, alpha_0


def gen_mask(im_origin: np.ndarray):
    # This assume generated text is within Center 256x256 crop
    no_logo = np.random.randint(1, 4)
    alpha_0 = np.zeros((512, 512))
    for _ in range(no_logo):
        im_origin, alpha_0 = add_logo(im_origin, alpha_0)

    num_effective_pixel = len(np.where(alpha_0 > 0)[0])
    return im_origin, num_effective_pixel


if __name__ == '__main__':
    src = cv2.imread("asset/image.jpg")

    for _ in range(10):
        res, num_pixels = gen_mask(src)
        print(f"num_effective_pixel: {num_pixels}")
        cv2.imshow("Result", res)
        cv2.waitKey(0)
