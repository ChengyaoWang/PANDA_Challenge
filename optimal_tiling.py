'''
    Algorithm to Compute "Optimal" Corrdinates for patches / tiles
    
    Update Aug 21st, 2020:
        Decouple with local environments
        Makes it more compatible with PyTorch Dataset Wrapper
'''
import cv2, skimage.io, math, warnings, os
import numpy as np 
import pandas as pd
import multiprocessing


class Config:
    # Which resolution level to extract from {0, 1, 2}
    slide_resolution_lvl = 1
    # Applied on Slides, equal to Patch sizes to make sure no tissue is left out
    pad_len = 256
    # Slide Resizing
    slide_resize_ratio = 1
    # Morphological Operation for Cleanning Masks
    dilation_kernel_size = (7, 7)
    # Hard Threshold for determining BG / Tissue 
    gray_threshold_mask_gen = 220
    # Dimentions of Patches
    patch_size = 256
    '''           
        min_patch_info:         minimum information required information in every patch
        min_axis_info:          minimum information required along the x / y dimension within a "patch_size" wide belt
        min_consec_axis_info:   Minimum Consecutive x / y on - bits
        min_decimal_keep:       Threshold fo rdecimal point for removing excessive patch
    '''
    min_patch_info = 0.35
    min_axis_info = 0.35
    min_consec_info = 0.35
    min_decimal_keep = 0.7
    sample_size = 36


class opt_tiling(Config):

    def __init__(self, data_csv, data_path):
        super().__init__()
        self.__dict__ = super().__dict__
        self.data_csv = data_csv
        self.data_path = data_path

    def __len__(self):
        return len(self.data_csv.index)

    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError(f'Valid Index Range 0 ~ {self.__len__() - 1}')    
        img = self.read_image(i)
        img, coords = self.locate_tiles(img)
        return self.yield_stack_patches(img, coords)
        
    '''
        Utility Functions:
            Image Padding
            Image Transposing
            Mask Tissue In Image
                Use Grey-Scaled Image, dilation kernels and gap-filling
            Determine if there're multiple tissue parts
            Locate the Patches of Interest
            Slicing out Patches from original image
    '''
    def _pad_image(self, image, pad_val):
        pad_len = self.pad_len
        if image is None:
            return None
        elif image.ndim == 2:
            return np.pad(image, ((pad_len, pad_len), (pad_len, pad_len)), pad_val)
        elif image.ndim == 3:
            return np.pad(image, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), pad_val)
        return None
    def _transpose_image(self, image):
        if image is None:
            return None
        elif image.ndim == 2:
            return np.transpose(image, (1, 0)).copy()
        elif image.ndim == 3:
            return np.transpose(image, (1, 0, 2)).copy()
        return None
    def _mask_tissue(self, image):
        # Elliptic Kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.dilation_kernel_size)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Mask the gray-scaled image (capturing tissue in Biopsy)
        mask = np.where(gray < self.gray_threshold_mask_gen, 1, 0).astype(np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)
        contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(mask, [cnt], 0, 1, -1)
        return mask
    def _get_tissue_parts_indices(self, tissue):
        split_points = np.where(np.diff(tissue) != 1)[0] + 1
        tissue_parts = np.split(tissue, split_points)
        return [tp for tp in tissue_parts if len(tp) >= (self.min_consec_info * self.patch_size)]
    def _get_tissue_subparts_coords(self, subtissue):
        start, end = subtissue[0], subtissue[-1]
        num_subparts = (end - start) / self.patch_size
        if num_subparts % 1 < self.min_decimal_keep and num_subparts >= 1:
            num_subparts = math.floor(num_subparts)
        else:
            num_subparts = math.ceil(num_subparts)
        shift = ((num_subparts * self.patch_size) - (end - start)) // 2
        return [i * self.patch_size + start - shift for i in range(num_subparts)]
    def _eval_and_append_xy_coords(self, coords, image, mask, x, y, transposed):
        patch_id = image[y: y + self.patch_size, x: x + self.patch_size, :].mean(axis = 2).reshape(-1)
        idx_tissue = np.where(patch_id <= 210)[0]
        idx_black = np.where(patch_id < 5)[0]
        idx_background = np.where(patch_id > 210)[0]
        if len(idx_tissue > 0):
            patch_id[idx_black] = 210
            patch_id[idx_background] = 210
            val1, val2 = int(patch_id.mean()), mask[y: y + self.patch_size, x: x + self.patch_size].mean()
            if val2 > self.min_patch_info:
                coords = np.concatenate([coords, [[val1, y, x]]])
        return coords    
    '''
        Filters out or Add Excessive / Lacking coordinates
    '''
    def _filtering_or_filling_coords(self, coords):
        # Sorting According to Information (Smaller Better)
        coords = np.array(sorted(coords, key = lambda x: x[0], reverse = False))
        if coords.shape[0] < self.sample_size:
            coords = np.tile(coords, reps = (math.ceil(1.0 * self.sample_size / coords.shape[0]), 1))
        coords = coords[: self.sample_size]
        return coords

    '''
        Read Raw Tiff images at certain resolution level
        Input: Index on self.data_csv file
        Output: np.ndarray
    '''
    def read_image(self, image_idx: int):
        image_path = self.data_path + self.data_csv.image_id[image_idx] + '.tiff'
        image_level_x = skimage.io.MultiImage(image_path)[self.slide_resolution_lvl]
        if self.slide_resize_ratio != 1:
            new_w = int(image_level_x.shape[1] * self.slide_resize_ratio)
            new_h = int(image_level_x.shape[0] * self.slide_resize_ratio)
            image_level_x = cv2.resize(image_level_x, (new_w, new_h), interpolation = cv2.INTER_AREA)
        return image_level_x

    '''
        Locating Optimal Tiles, Return it's coordinates (Used Directly for Extraction)
        Input: Input Image -> np.ndarray
        Output: Coordinates -> np.ndarray
    '''
    def locate_tiles(self, image: np.ndarray):

        mask = self._mask_tissue(image)
        coords = np.zeros([0, 3], dtype = int)
        
        # Padding to make sure no tissues are left outside
        image = self._pad_image(image, 'maximum')
        mask = self._pad_image(mask, 'minimum')

        # Collect information along x / y axis
        y_sum = mask.sum(axis = 1)
        x_sum = mask.sum(axis = 0)

        # Transpose the image if tissue is possibly horizontally
        if len(np.where(x_sum > 0)[0]) > len(np.where(y_sum > 0)[0]):
            image = self._transpose_image(image)
            mask = self._transpose_image(mask)
            y_sum, _ = x_sum, y_sum
            transposed = True
        else:
            transposed = False

        # Tissue locations with much number of on - bits
        y_tissue = np.where(y_sum >= (self.patch_size * self.min_axis_info))[0]
        if len(y_tissue) < 1:
            warnings.warn("Not Enough Tissue in Image (y-dim)", RuntimeWarning)
            return image, np.array([0, 0, 0])

        # Slice the Slides along y-axis if tissue is discontinuous
        y_tissue_parts_indices = self._get_tissue_parts_indices(y_tissue)
        if len(y_tissue_parts_indices) < 1:
            warnings.warn("Not Enough Tissue in Image (y-dim", RuntimeWarning)
            return image, np.array([0, 0, 0])

        # Loop over the tissues in y-dimension
        for y_idx in y_tissue_parts_indices:
            y_tissue_subparts_indices = self._get_tissue_subparts_coords(y_idx)
            for y in y_tissue_subparts_indices:
                x_slice_num = mask[y: y + self.patch_size, :].sum(axis = 0)
                x_tissue = np.where(x_slice_num >= (self.patch_size * self.min_axis_info))[0]
                x_tissue_parts_indices = self._get_tissue_parts_indices(x_tissue)
                
                # Loop over the tissues in x_dimension
                for x_idx in x_tissue_parts_indices:
                    x_tissue_subparts_coords = self._get_tissue_subparts_coords(x_idx)
                    for x in x_tissue_subparts_coords:
                        coords = self._eval_and_append_xy_coords(coords, image, mask, x, y, transposed)

        if len(coords) < 1:
            warnings.warn('Not Enough Tissue in Image (x - dim', RuntimeWarning)
            return image, np.array([0, 0, 0])

        return image, np.array(coords)


    def yield_concat_patches(self, image, coords, filtering = True):
        if not filtering:
            raise ValueError('Please Enable filtering to generate images with same resolution')
        l = int(math.sqrt(self.sample_size))
        if filtering:
            coords = self._filtering_or_filling_coords(coords)
        retImg = np.empty((self.sample_size, self.patch_size, self.patch_size, 3), dtype = int)
        for i, (v, y, x) in enumerate(coords):
            retImg[i] = image[y: y + self.patch_size, x: x + self.patch_size, :].copy()
        retImg = cv2.hconcat([
            cv2.vconcat([retImg[v + h * l] for v in range(l)]) for h in range(l)
        ])
        return retImg

    def yield_stack_patches(self, image, coords, filtering = True):
        retImg = []
        if filtering:
            coords = self._filtering_or_filling_coords(coords)
        for i, (v, y, x) in enumerate(coords):
            retImg.append(image[y: y + self.patch_size, x: x + self.patch_size, :].copy())
        return np.array(retImg)


'''
    Parallel Extraction
'''

# Slave Function
def jobs(idx_q, mp_lock, data_csv, data_path: str, save: bool):

    mp_lock.acquire()
    print(f'Process: {os.getpid()} Started')
    mp_lock.release()

    # Instantiate Tile Extractor
    tile_extractor = opt_tiling(data_csv = data_csv, data_path = data_path)

    # If there're any task left
    while not idx_q.empty():

        mp_lock.acquire()
        curIdx = idx_q.get(0)
        print(curIdx, end = ' ')
        mp_lock.release()
        
        # Extract Patches
        stacked_patches = tile_extractor[curIdx]

        # Output HandShake
        mp_lock.acquire()
        print(f'Output Dims: {stacked_patches.shape}')
        mp_lock.release()
        if save:
            name = tile_extractor.data_csv.iloc[curIdx].image_id
            np.save(data_path + 'npy/' + name, stacked_patches)
    
    return 0

# Master Function
def parallel_fetching(indexRange, data_csv, data_path: str, n_jobs: int = 4, save: bool = True):
    
    idx_q = multiprocessing.Queue()
    mp_lock = multiprocessing.Lock()

    for i in indexRange:
        idx_q.put(i)

    pool = [
        multiprocessing.Process(
            target = jobs,
            args = (
                idx_q, mp_lock, data_csv, data_path, True, 
            )
        ) for _ in range(n_jobs)
    ]
    for p in pool:
        p.start()
    for p in pool:
        p.join()
    
    print(f'All Completed')





if __name__ == '__main__':

    '''
        Parallel Way of Extraction
    '''

    data_csv = pd.read_csv('/mnt/chengyao/prostate-cancer-grade-assessment/train.csv')
    data_path = '/mnt/chengyao/prostate-cancer-grade-assessment/train_images/'
    total_num = len(data_csv.index)

    parallel_fetching(
        indexRange = range(total_num),
        data_path = data_path,
        data_csv = data_csv,
        n_jobs = 16,
        save = True
    )

    '''
        Doing them one-by-one
    '''

    # tile_extractor = opt_tiling()
    # stacked_patches = tile_extractor[0]
    # print(f'Dimensions for Extracted Patches: {stacked_patches.shape}')
