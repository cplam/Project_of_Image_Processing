import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

INPUT_DIR = "./output/output_member1"
OUTPUT_DIR = "./output/output_member2"
KERNEL_SIZE = 3
PRESERVE_EYES = True  # Set to False to use aggressive filtering

class MorphologyRefiner:
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size
        self.se = np.zeros((kernel_size, kernel_size), dtype=bool)
        center = kernel_size // 2
        self.se[center, :] = True
        self.se[:, center] = True
    
    def erode(self, mask: np.ndarray) -> np.ndarray:
        binary = mask > 127
        eroded = ndimage.binary_erosion(binary, structure=self.se)
        return (eroded * 255).astype(np.uint8)
    
    def dilate(self, mask: np.ndarray) -> np.ndarray:
        binary = mask > 127
        dilated = ndimage.binary_dilation(binary, structure=self.se)
        return (dilated * 255).astype(np.uint8)
    
    def opening(self, mask: np.ndarray) -> np.ndarray:
        return self.dilate(self.erode(mask))
    
    def closing(self, mask: np.ndarray) -> np.ndarray:
        return self.erode(self.dilate(mask))
    
    def remove_small_objects(self, mask: np.ndarray, min_size: int = 500) -> np.ndarray:
        """Remove connected components smaller than min_size"""
        binary = mask > 127
        labeled, num_features = ndimage.label(binary)
        
        if num_features == 0:
            return mask
        
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        
        result = np.zeros_like(binary)
        for i, size in enumerate(sizes, start=1):
            if size >= min_size:
                result[labeled == i] = True
        
        return (result * 255).astype(np.uint8)
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        binary = mask > 127
        filled = ndimage.binary_fill_holes(binary)
        return (filled * 255).astype(np.uint8)
    
    def refine(self, mask_from_member1: np.ndarray) -> np.ndarray:
        """Process Member 1's mask with eye preservation"""
        step1 = self.opening(mask_from_member1)
        step2 = self.closing(step1)
        
        if PRESERVE_EYES:
            # Use smaller threshold for eyes
            h, w = mask_from_member1.shape
            eye_min_size = max(30, int((h * w) * 0.0005))  # 0.05% of image area
            step3 = self.remove_small_objects(step2, min_size=eye_min_size)
        else:
            step3 = self.remove_small_objects(step2, min_size=500)
        
        step4 = self.fill_holes(step3)
        return step4

# Rest of the code remains the same...
