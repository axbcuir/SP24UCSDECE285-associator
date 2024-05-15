from utils import *
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

image = cv2.imread('images/room/4.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks, alpha=1)
plt.axis('off')
plt.savefig('images/room/4_mask.png')
