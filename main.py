import matplotlib.pyplot as plt

from utils import *
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import clip
from associator import *

device = "cuda"

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

associator = Associator(sam, clip_model, preprocess)

img_path = 'images/a'

imgs = read_images_from_directory(img_path)
for img in imgs[:5]:
    associator.add_img(img)

target = associator.cutout_region(associator.imgs[0]['img'], associator.imgs[0]['segments'][39]['mask'])

# show the target
plt.imshow(target)
plt.show()

# find associated objects
associated_cutouts = associator.query(target)

show_images_grid(associated_cutouts)
