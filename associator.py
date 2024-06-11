from utils import *
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from scipy.spatial import KDTree
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt

class Associator():

    def __init__(self, sam, clip, clip_preprocess, image_path=None):
        self.sam = sam
        self.clip = clip
        self.preprocess = clip_preprocess
        self.device = 'cuda'
        if image_path:
          with open(image_path, 'rb') as f:
            self.imgs = pickle.load(f)
        else:
          self.imgs = []
    
    def save_img(self, image_path):
      with open(image_path, 'wb') as f:
        pickle.dump(self.imgs, f)

    def add_img(self, img):
        masks = self.generate_masks(img)
        embeddings = self.embed_image(img, masks)
        img_info = {
            'img': img,
            'segments': embeddings
        }
        self.imgs.append(img_info)

    def generate_masks(self, img):
        self.sam.to(device='cuda')
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        masks = mask_generator.generate(img)
        self.sam.to(device='cpu')   # save VRAM
        torch.cuda.empty_cache()

        return masks

    def cutout_region(self, img, mask):
        """
        Given image and mask from SAM, cut out the masked region
        :param img: np.ndarray
        :param mask: dict
        :return res: np.ndarray
        """
        assert img.shape[:2] == mask['segmentation'].shape[:2]
        msk = mask['segmentation'].astype(np.uint8)
        res = cv2.bitwise_and(img, img, mask=msk)
        bbox_start_x = mask['bbox'][0]
        bbox_start_y = mask['bbox'][1]
        bbox_w = mask['bbox'][2]
        bbox_h = mask['bbox'][3]
        res = res[bbox_start_y:(bbox_start_y + bbox_h), bbox_start_x:(bbox_start_x + bbox_w), :]

        return np.uint8(res)

    def clip_encode(self, img):
        pil_img = Image.fromarray(img)
        embedding = self.clip.encode_image(self.preprocess(pil_img).unsqueeze(0).to(self.device))
        return embedding.cpu().numpy().flatten()

    def embed_image(self, img, masks):
        segments = []
        for mask in masks:
            cutout = self.cutout_region(img, mask)
            with torch.no_grad():
                embedding = self.clip_encode(cutout)
            segment_descriptor = {
                'mask': mask,
                'embedding': embedding
            }
            segments.append(segment_descriptor)

        torch.cuda.empty_cache()    # save VRAM
        return segments

    def inspect_segment(self, img_idx, segment_idx):
        img = self.imgs[img_idx]['img']
        mask = self.imgs[img_idx]['segments'][segment_idx]['mask']
        cutout = self.cutout_region(img, mask)
        plt.imshow(cutout)
        plt.show()
    
    def find_largest_inclusive_mask(self, target_mask, mask_list, overlap_threshold=0.9):
        """
        Finds the largest mask in a list that overlaps with the target mask by at least a specified percentage.

        Parameters:
        - target_mask (np.array): The target mask as a numpy array with elements 0 or 1.
        - mask_list (list of np.array): List of masks, each an numpy array formatted like the target_mask.
        - overlap_threshold (float): The minimum overlap percentage required (between 0 and 1).

        Returns:
        - largest_mask (np.array): The largest mask that includes the target mask.
        - largest_mask_index (int): Index of the found mask in the list.
        """
        largest_area = 0
        largest_mask = None
        largest_mask_index = -1

        target_area = np.sum(target_mask)

        n = 0
        for index, mask in enumerate(mask_list):
            # Calculate the intersection
            intersection = np.logical_and(target_mask, mask)
            intersection_area = np.sum(intersection)

            # Calculate the overlap percentage
            overlap_percentage = intersection_area / target_area

            # Check if the overlap is above the threshold
            if overlap_percentage >= overlap_threshold:
                n += 1
                current_area = np.sum(mask)
                if current_area > largest_area:
                    largest_area = current_area
                    largest_mask = mask
                    largest_mask_index = index

        return largest_mask, largest_mask_index
        # return n
    
    def test(self, img):
        """
        Function for Test some utils of Associator.
        """
        with torch.no_grad():
              query_point = self.clip_encode(img)
        img = self.imgs[6]   # second image
        embedding_list = np.array([segment['embedding'] for segment in img['segments']])
        tree = KDTree(embedding_list)
        _, idx = tree.query(query_point)
        mask = img['segments'][idx]['mask']
        msk = mask['segmentation'].astype(np.uint8)

        # Find Largest Inclusive Mask
        msk_list = np.array([segment['mask']['segmentation'].astype(np.uint8) for segment in img['segments']])
        largest_mask, largest_mask_index = self.find_largest_inclusive_mask(msk, msk_list)

        return self.cutout_region(img['img'], img['segments'][largest_mask_index]['mask'])
        # return self.find_largest_inclusive_mask(msk, msk_list)

    def find_MaxTarget(self, img, mask):
        msk = mask['segmentation'].astype(np.uint8)
        msk_list = np.array([segment['mask']['segmentation'].astype(np.uint8) for segment in img['segments']])
        largest_mask, largest_mask_index = self.find_largest_inclusive_mask(msk, msk_list)

        return self.cutout_region(img['img'], img['segments'][largest_mask_index]['mask'])

    def query(self, img):
        """
        Given an image of an object, find the same object in already seen images
        :param img: np.ndarray
        :return: res: list of associated image cutouts
        """
        with torch.no_grad():
            query_point = self.clip_encode(img)

        res = []
        for img in self.imgs:
            embedding_list = np.array([segment['embedding'] for segment in img['segments']])
            tree = KDTree(embedding_list)
            _, idx = tree.query(query_point)

            # find the largest inclusive part
            mask = img['segments'][idx]['mask']
            msk = mask['segmentation'].astype(np.uint8)

            msk_list = np.array([segment['mask']['segmentation'].astype(np.uint8) for segment in img['segments']])
            largest_mask, largest_mask_index = self.find_largest_inclusive_mask(msk, msk_list)

            res.append(self.cutout_region(img['img'], img['segments'][largest_mask_index]['mask']))

        torch.cuda.empty_cache()  # save VRAM
        return res
