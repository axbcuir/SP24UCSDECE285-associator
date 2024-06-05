from utils import *
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from scipy.spatial import KDTree
from PIL import Image
import pickle
import numpy as np
from tree import *

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
        segment_trees = self.embed_image(img, masks)
        img_info = {
            'img': img,
            'segments': segment_trees
        }
        print("Added an image.")
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

        # sort the segments by decending area
        segments.sort(key=lambda x: x['mask']['area'], reverse=True)

        return trees_from_sorted_list(segments, self.is_child)

    def is_child(self, proposed_child, proposed_parent, overlap_threshold=0.9):
        """
        Determine if a mask is a subset of another mask
        :param proposed_child: dict
        :param proposed_parent: dict
        :param overlap_threshold: float
        :return: bool
        """
        # Calculate the intersection
        intersection = np.logical_and(proposed_child['mask']['segmentation'], proposed_parent['mask']['segmentation'])
        intersection_area = np.sum(intersection)
        # Calculate the overlap percentage
        child_area = proposed_child['mask']['area']
        overlap_percentage = intersection_area / child_area

        return overlap_percentage >= overlap_threshold

    def find_closest(self, query, cloud):
        """
        Given a query point and cloud, find the closest point on the cloud.
        :param query: np.ndarray
        :param cloud: np.ndarray
        :param mode: 'euclidean' or 'cosine'
        :return: idx: int, high_match: bool
        """
        high_match = False
        tree = KDTree(cloud)
        dists, idxs = tree.query(query, k=10)
        # empirical rule to judge whether there actually is a matching segment
        differences = [dists[i + 1] - dists[i] for i in range(len(dists) - 1)]
        criterion = np.max(differences)
        print(criterion)
        idx = idxs[0]
        if criterion < 1:
            high_match = False
        else:
            high_match = True

        return idx, high_match

    def query(self, img):
        """
        Given an image of an object, find the same object in already seen images
        :param img: np.ndarray
        :param mode: 'euclidean' or 'cosine'
        :return: res: list of associated image cutouts
        """
        with torch.no_grad():
            query_point = self.clip_encode(img)
        res = []
        for seen_img in self.imgs:
            embedding_list = np.array([a['embedding'] for a in nodes_to_list(seen_img['segments'])])
            idx, success = self.find_closest(query_point, embedding_list)

            # find the largest inclusive part
            root_idx = seen_img['segments'].index(seen_img['segments'][idx].get_root())
            root_mask = seen_img['segments'][root_idx].get_value()['mask']
            result = self.cutout_region(seen_img['img'], root_mask)
            if success:
                res.append(result)
            else:
                res.append(none_img(result))

        torch.cuda.empty_cache()  # save VRAM
        return res

    def visualize_segments(self, img_idx, seg_idx_start, seg_idx_end):
        """
        Visualize segments from an image
        :param img_idx: int
        :param seg_idx_start: int
        :param seg_idx_end: int
        """
        cutouts = []
        num_of_segments = len(self.imgs[img_idx]['segments'])
        if seg_idx_end > num_of_segments:
            seg_idx_end = num_of_segments
        if seg_idx_end - seg_idx_start > 36:
            raise ValueError('Too many to visualize')
        if seg_idx_start >= seg_idx_end:
            raise ValueError('Segment idx start >= segment idx end')
        for i in range(seg_idx_start, seg_idx_end):
            mask = self.imgs[img_idx]['segments'][i].get_value()['mask']
            cutout = self.cutout_region(self.imgs[img_idx]['img'], mask)
            cutouts.append(cutout)
        show_images_grid(cutouts)
