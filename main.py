import matplotlib.pyplot as plt

from utils import *
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import clip
import pickle
from associator import *

device = "cuda"

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

associator_path = 'associators/mytable_all.pkl'

associator = Associator(sam, clip_model, preprocess)

imgs = read_images_from_directory('images/a') + read_images_from_directory('images/b')
for img in imgs:
    associator.add_img(img)

# save the associator object, since segmenting and embedding images take long time
with open(associator_path, 'wb') as f:
    pickle.dump(associator, f)

# with open(associator_path, 'rb') as f:
#     associator = pickle.load(f)


# Visualize some of the segments of the first image. Choose one as the target.
cutouts = []
for i in range(36, 72):
    cutout = associator.cutout_region(associator.imgs[0]['img'], associator.imgs[0]['segments'][i]['mask'])
    cutouts.append(cutout)
show_images_grid(cutouts)

target = associator.cutout_region(associator.imgs[0]['img'], associator.imgs[0]['segments'][47]['mask'])

# show the target
plt.imshow(target)
plt.show()

# find associated objects
associated_cutouts = associator.query(target)

# Visualize results
show_images_grid(associated_cutouts)
