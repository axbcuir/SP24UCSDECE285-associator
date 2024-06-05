from associator import *

device = "cuda"

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

img_path = 'imgs/mytable_new.pkl'

# associator = Associator(sam, clip_model, preprocess)
# imgs = read_images_from_directory('images/a') + read_images_from_directory('images/b')
# for img in imgs:
#     associator.add_img(img)
# # save the associator object's images
# associator.save_img(img_path)

associator = Associator(sam, clip_model, preprocess, img_path)

# Visualize some of the segments of the first image. Choose one as the target.
associator.visualize_segments(0, 36, 72)

target_mask = associator.imgs[0]['segments'][41].get_value()['mask']
target = associator.cutout_region(associator.imgs[0]['img'], target_mask)

# show the target
plt.imshow(target)
plt.show()

# find associated objects
associated_cutouts = associator.query(target)

# Visualize results
show_images_grid(associated_cutouts)

# associator.visualize_segments(16, 0, 36)
#
# associator.add_img(target)
# associator.visualize_segments(20, 0, 36)
