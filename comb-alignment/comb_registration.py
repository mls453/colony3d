import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def keypoints_match_image(keypoints_file, image_file):
    """confirm that keypoints correspond to image."""
    keypoint_name = os.path.splitext(
        os.path.basename(keypoints_file)
    )[0]
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    return keypoint_name == image_name

def get_upper_keypoints(keypoints):
    """ Just get the keypoints on the top of the frame."""
    keypoints = np.concatenate([keypoints[0:2],
                               keypoints[-4:]]
                              )
    return keypoints

def show_keypoints_on_image(image, keypoints, figsize=(5,5)):
    """ Display keypoints on image.
    
    Args:
        image: 2D or 3D (HxWxC) array 
        keypoints: keypoints array (nx2)
        figsize: size to display image
        
    """
    
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1])
    
def get_warp_matrix(keypoints, reference_keypoints, 
                    return_inliers=False, verbose=True):
    """ Get the matrix needed to warp keypoints onto the reference keypoints.
    
    Args:
        keypoints: nx2
        reference_keypoints: nx2
        return_inliers: if True return inliers along with warp matrix
        verbose: if True print warning when fewwer than 5 inliers
    """
    
    transform, inliers = cv2.estimateAffinePartial2D(keypoints, 
                                                     reference_keypoints,
                                                     ransacReprojThreshold=50,
                                                     confidence=.98
                                                     )
    if verbose:
        if np.sum(inliers) <= 4:
            print(f"Warning, only {np.sum(inliers)} inliers.")
            
    if return_inliers:
        return transform, inliers
    
    return transform
    
    
def get_frames_not_to_warp():
    """ Old big german frames with keypoints out of image."""
    return ['DSC_8294', 'DSC_8295', 'DSC_8298', 
            'DSC_8302', 'DSC_8304'
           ]


def load_reference_mask(nest_photos_folder, resize=False):
    """ Load mask that all other frames are aligned to.
    
    Args:
        nest_photos_folder: full path to the nest_photos folder
        resize: if True make same size as raw image
    
    Returns:
        2D mask image
    """
    reference_mask_file = os.path.join(nest_photos_folder, 
                                       "CC1", "20210615", 
                                       "masks", "DSC_2500.png"
                                      )
    reference_mask = cv2.imread(reference_mask_file, 
                                cv2.IMREAD_GRAYSCALE
                               )
    if resize:
        rescale_factor = 1 / .15
        reference_mask = cv2.resize(reference_mask, None,
                                    fx=rescale_factor,
                                    fy=rescale_factor,
                                    interpolation=cv2.INTER_LINEAR
                                   )
    return reference_mask
        
    
def load_reference_image(nest_photos_folder, resize=False):
    """ Load image that all other frames are aligned to.
    
    Args:
        nest_photos_folder: full path to the nest_photos folder
        resize: if True make same size as raw mask
    
    Returns:
        3D image
    """
    image_file = os.path.join(nest_photos_folder, 
                              "CC1", "20210615", 
                              "DSC_2500.JPG"
                             )
    image = plt.imread(image_file)
    if resize:
        image = cv2.resize(image, None,
                            fx=.15,
                            fy=.15,
                            interpolation=cv2.INTER_AREA
                           )
    return image
   
def load_keypoints(keypoints_file, resize=False, 
                   return_upper=False
                  ):   
    """ Load the keypoint file for a frame.
    Args:
        keypoints_file: full path to the keypoint file
        resize: if True scale keypoints for raw image size 
            instead of the size of the comb/wood/background mask
        return_upper: if True, only return upper frame keypoints
    Return:
        keypoint array
    """
    keypoints = np.genfromtxt(keypoints_file, delimiter=',')
    if resize:
        keypoints /= .15
        
    if return_upper:
        keypoints = get_upper_keypoints(keypoints)
    return keypoints
    
def load_reference_keypoints(nest_photos_folder, resize=False, 
                             return_upper=False
                            ):
    """ Load the keypoints that correspond to the reference image.
    
    Args:
        nest_photos_folder: full path to the nest_photos folder
        resize: if True scale keypoints for raw image size instead
            of the size of the comb/wood/background mask
        return_upper: if True, only return upper frame keypoints
    Return:
        keypoint array
    """
    
    keypoints_file = os.path.join(nest_photos_folder, 
                                 "CC1", "20210615", 
                                 "keypoints", "DSC_2500.csv"
                                 )
    keypoints = load_keypoints(keypoints_file, resize, 
                               return_upper
                              )
    return keypoints

def mirror_keypoints_horizontal(keypoints, image_width):
    """ Horizontally mirror keypoints in frame image.
    
    Args:
        keypoints: nx2
        image_width: width of image keypoints come from
    Return:
        keypoints: nx2
    """
    
    if not isinstance(keypoints, np.ndarray):
        raise RuntimeError(
            f"Keypoints should be stored in numpy array"
        )
        
    keypoints[:, 0] = image_width - keypoints[:, 0]
    keypoints = np.concatenate([keypoints[3:], keypoints[:3]])
    
    return keypoints
    
    