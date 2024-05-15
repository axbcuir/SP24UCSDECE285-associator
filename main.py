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

room1 = cv2.imread('images/room/1.jpg')
room2 = cv2.imread('images/room/2.jpg')
room3 = cv2.imread('images/room/3.jpg')
room4 = cv2.imread('images/room/4.jpg')
room5 = cv2.imread('images/room/5.jpg')

imgs = [room1, room2, room3, room4, room5]
for img in imgs:
    associator.add_img(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# the target should be the vase
target = associator.cutout_region(associator.imgs[0]['img'], associator.imgs[0]['segments'][10]['mask'])

# show the target
plt.imshow(target)
plt.show()

# find associated objects
associated_cutouts = associator.query(target)

# this one is good
plt.imshow(associated_cutouts[2])
plt.show()

# this one is bad
plt.imshow(associated_cutouts[4])
plt.show()
