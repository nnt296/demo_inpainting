import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data
from os import listdir

from PIL import Image

from mask_helper import gen_mask
from utils.tools import default_loader, is_image_file, normalize
import os

import torchvision.transforms as transforms


class SquarePad:
    def __init__(self, value=0):
        self.value = value

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        lp = int((max_wh - w) / 2)
        tp = int((max_wh - h) / 2)
        result = Image.new(image.mode, (max_wh, max_wh), self.value)
        result.paste(image, (lp, tp))
        return result


class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape, with_subfolder=False, random_crop=True, return_name=False):
        super(Dataset, self).__init__()
        if with_subfolder:
            self.samples = self._find_samples_in_subfolders(data_path)
        else:
            self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name

        self.pad_square = SquarePad()
        self.do_resize = transforms.Resize(512)

        self.transform = transforms.Compose([
            # Padding to max_size (512)
            # SquarePad(),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.samples[index])
        img = default_loader(path)
        img = self.pad_square(img)
        img = self.do_resize(img)

        cv_im = np.array(img)
        cv_im = cv2.cvtColor(cv_im, cv2.COLOR_RGB2BGR)
        gen, mask, bbox = gen_mask(cv_im)
        mask = mask.astype(np.float32)
        # Center crop mask
        mask = mask[128:128 + 256, 128:128 + 256]
        mask = torch.from_numpy(mask).unsqueeze(0)

        img = self.transform(img)  # pad, crop & turn the image to a tensor
        img = normalize(img)

        gen = cv2.cvtColor(gen, cv2.COLOR_BGR2RGB)
        gen = Image.fromarray(gen)
        gen = self.transform(gen)
        gen = normalize(gen)

        if self.return_name:
            return self.samples[index], img
        else:
            return gen, mask, bbox, img

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)
