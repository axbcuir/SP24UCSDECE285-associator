from utils import *
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from scipy.spatial import KDTree
from PIL import Image


class Associator():

    def __init__(self, sam, clip, clip_preprocess):
        self.sam = sam
        self.clip = clip
        self.preprocess = clip_preprocess
        self.device = 'cuda'
        self.imgs = []

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
            res.append(self.cutout_region(img['img'], img['segments'][idx]['mask']))

        torch.cuda.empty_cache()  # save VRAM
        return res
