import os
import cv2
import random

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from mask_helper import gen_mask


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])

        left = int((max_wh - w) / 2)
        top = int((max_wh - h) / 2)

        result = Image.new(image.mode, (max_wh, max_wh), 0)
        result.paste(image, (left, top))

        return result


class FoodDataset(Dataset):
    def __init__(self, img_dir, im_names_path):
        self.img_dir = img_dir
        self.im_names_path = im_names_path

        if not os.path.isdir(self.img_dir):
            raise RuntimeError(f"Invalid img_dir: {img_dir}")
        if not os.path.isfile(self.im_names_path):
            raise RuntimeError(f"Invalid im_names_path: {im_names_path}")

        self.image_paths = []

        self.read_images()

        self.basic_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(512)
        ])

        self.flip = transforms.RandomHorizontalFlip(p=1.1)

        self.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def read_images(self):
        with open(self.im_names_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                image_path = os.path.join(self.img_dir, line + ".jpg")
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # Pad square & resize 512
        img = self.basic_transform(img)

        im_np = np.array(img)
        # Convert to opencv BGR format
        im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
        masked_im_np, num_effective_pixels = gen_mask(im_np)

        # Convert to PIL RGB format
        masked_im_np = cv2.cvtColor(masked_im_np, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        masked_im = Image.fromarray(masked_im_np)

        # Training/Test transform
        if random.uniform(0, 1) < 0.5:
            masked_im = self.flip(masked_im)
            img = self.flip(img)

        masked_im = self.transform(masked_im)
        img = self.transform(img)

        return img, masked_im, num_effective_pixels


if __name__ == '__main__':
    import cv2

    ds = FoodDataset(img_dir="/mnt/CVProjects/Cuong/food101/images",
                     im_names_path="/mnt/CVProjects/Cuong/food101/meta/meta/test.txt")

    loader = DataLoader(ds, batch_size=2, shuffle=True)

    for _, m, n in loader:
        print("NUM: ", n)
        print(m.shape)
        break
