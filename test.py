from PIL import Image, ImageEnhance
import numpy as np
import cv2
from mask_helper import gen_mask
from torchvision import transforms
#img = Image.open(r"C:\Users\ffleader1\PycharmProjects\cloud_function_remove_watermark\tmp\c6477dd0-8004-4485-b27c-fcbbf8c2-a469075e-201027190334.jpg").convert("RGB")

img = Image.open(r"C:\Users\ffleader1\Downloads\4c7f2e97-dfba-45f6-d939-f3c913b98411.jpg").convert("RGB")
img = img.resize((512, 512), Image.ANTIALIAS)

im_np = np.array(img)
# Convert to opencv BGR format
im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
masked_im_np, num_effective_pixels = gen_mask(im_np)
print("value: ", len(np.where(masked_im_np > 155)[0]))
# Convert to PIL RGB format
masked_im_np = cv2.cvtColor(masked_im_np, cv2.COLOR_BGR2RGB)

# Convert to PIL
masked_im = Image.fromarray(masked_im_np)
# masked_im = ImageEnhance.Contrast(masked_im).enhance(0.628)
# img = ImageEnhance.Contrast(img).enhance(0.628)

masked_im.show()

img.show()

trans = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

print( np.asarray([594,349]).astype(np.uint8))
# masked_im2 = trans(masked_im).unsqueeze(0)
# masked_im2.show()
# img2 = trans(img).unsqueeze(0)
# img2.show()

