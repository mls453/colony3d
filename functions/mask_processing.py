import numpy as np
import cv2


def get_class_contours(mask, class_id):
    """ Get contours for class_id in mask. 
    
    Args:
        mask: 2D numpy array 
        class_id: value of class of interest in mask
    Return:
        list of cv2 contours
    """
    class_mask = np.where(mask==class_id, 1, 0)
    class_mask = class_mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(class_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_NONE
                                          )
    
    return contours

def get_distance_to_class(point, mask, class_id):
    """ Return closest distance from point to class_id in mask and that postion.
    
    Args:
        point: (i, j) numpy indexing (from top left: (row, column))
        mask: 2D numpy array
        class_id: value of class of interest in mask
    
    Return:
        distance value and position if exists, otherwise returns False, False
    """
    if mask[point[0], point[1]] == class_id:
        return 0, point
    
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    if point.shape != (2,):
        raise RuntimeError(f"point should have shape (2,)"
                           f"but has shape {point.shape} instead."
                          )
    
    class_inds = np.argwhere(mask==class_id)
    if class_inds.size > 0:
        diffs = class_inds - point
        distances = np.linalg.norm(diffs, axis=1)
        closest_ind = np.argmin(distances)
        return distances[closest_ind], class_inds[closest_ind]
    
    return False, False

def get_interior_mask(content_mask, wood_class=1):
    """Return mask of all space within the wood frame in mask of comb contents.
    
    Args:
        content_mask: 2D array of comb contents
        wood_class: value of wood in content mask
        
    Return:
        2D mask of size content mask with 1 inside frame and 0 outside.
    """
    wood_contours = get_class_contours(content_mask, class_id=wood_class)
    interior_mask = np.zeros_like(content_mask, dtype=np.uint8)
    cv2.drawContours(interior_mask, wood_contours, 1, 1, -1)
    return interior_mask

def dilate_class(mask, class_id, kernel_size=5):
    """ Make blobs of class_id slightly larger with dialation.
    
    Sometimes there is a thin layer of false wood around the
    comb border that we may want to remove.
    
    mask: 2D numpy array
    class_id: value of class of interest in mask
    kernel_size: kernel size for dialation
    
    return copy of mask with dialation applied
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    class_mask = np.where(mask==class_id, 1, 0)
    class_mask = class_mask.astype(np.uint8)
    dilation = cv2.dilate(class_mask, kernel, iterations=1)
    dilated_mask = np.where(dilation, class_id, mask)
    return dilated_mask
    