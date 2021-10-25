import cv2
import numpy as np
import os
import glob



def get_tangent_slope(contour, ind, num_points):
    """ Get tangent slope at ind on contour.
        But don't actually return slope, instead return 
        dx and dy where (dy/dx) is the slope. This makes getting
        taking the inverse more stable and when vertical slope
    
    Args:
        contour: cv2 contour object
        ind: ind of point want tangent for
        num_points: how many points to average on each size when calculating  
                    tangent
    """

    lowwer_bound = ind-num_points
    if lowwer_bound < 0:
        point1 = np.concatenate([contour[lowwer_bound:, 0], contour[:ind, 0]])
    else:
        point1 = contour[lowwer_bound:ind, 0]
        
    upper_bound = ind+num_points+1
    if upper_bound >= contour.shape[0]:
        wrap_ind = upper_bound-contour.shape[0]
        point2 = np.concatenate([contour[ind:, 0], contour[:wrap_ind, 0]])
    else:
        point2 = contour[ind+1:upper_bound, 0]

    point1 = np.mean(point1, 0)
    point2 = np.mean(point2, 0)
    
    dif = point2 - point1
    
    norm_dif = dif / np.linalg.norm(dif)
    dif_x, dif_y = norm_dif[0], norm_dif[1]
    
    return dif_x, dif_y

def get_perpendicular_to_tangent_slope(contour, ind, num_points):
    """ Perpendicular slope is just negative inverse of tangent slope.
    But return dx and dy instead of actual slope.
    
    Args:
        contour: cv2 contour object
        ind: ind of point want tangent for
        num_points: how many points to average on each size
    """
    
    dx_tangent, dy_tangent = get_tangent_slope(contour, ind, num_points)
    
    dx = -dy_tangent
    dy = dx_tangent
    
    return dx, dy


def draw_line_with_slope(im, point, slope_dx, slope_dy, line_length, color, width, out=False):
    """ Draw line witth tangent_slope at point with length line_length.
    
    Args: 
        im: 2D or 3D numpy array
        point: where to center line (x,y)
        slope_dx: slope dx of line to draw
        slope_dy: slope dy of line to draw
        line_length: how long to draw the line
        color: color of line, number if im is 2D bgr if 3D
        width: width of line
        out: just draw the half of the line that goes in the positive direction
    """
    
    p1_x = int(point[0] - (line_length / 2) * slope_dx)
    p1_y = int(point[1] - (line_length / 2) * slope_dy)
    p2_x = int(point[0] + (line_length / 2) * slope_dx)
    p2_y = int(point[1] + (line_length / 2) * slope_dy)
    
    cv2.line(im, [p1_x, p1_y], [p2_x, p2_y], color, width)
    
    return im

def draw_perpendiculars_on_contour(im, contour, spacing, line_length, buffer):
    """ 
    Args:
        im: image to draw contours on
        contour: contour to draw perdindiculars on
        spacing: how often to draw a perpendicular around contour
        line_length: length of perpendiculars
        buffer: how many points to average one edge to calculate tangent
    """
    
    cv2.drawContours(im, [contour], -1, 4, 1)
    for ind in range(0, len(contour), spacing):
        slope_dx, slope_dy = get_perpendicular_to_tangent_slope(contour, ind, buffer)
        draw_line_with_slope(im, contour[ind, 0], 
                             slope_dx, slope_dy, 
                             line_length, 7, 1, out=False
                            )
        cv2.circle(im, contour[ind, 0], 1, 0, -1)
    
    return im

def is_point_in_target(mask, point, target, anti_target=False):
    """ Check if (x,y) point in mask has value == target.
    
    Args:
        mask: 2D array
        point: (x, y)
        target: value looking for in mask
    """
    mask_value = point_mask_value(mask, point)
    if  mask_value == target:
        if anti_target:
            return False
        return True
    else:
        if anti_target:
            return True
        return False

def point_mask_value(mask, point):
    """ Which value in mask is at point.
    
    Args:
        mask: 2D array
        point: (x, y)
        
    Return False if point is not in mask
    """
    
    if not is_point_in_mask(mask, point):
        return False

    i = int(point[1]) # maybe sometimes should be mask.shape[0] - point[1]
    j = int(point[0])
    return mask[i, j]
    
def is_point_in_mask(mask, point):
    """ Check if (x,y) point is in mask.
    
    Args:
        mask: 2D array
        point: (x, y)
    """
    
    if type(point) is list:
        assert len(point) == 2, f"point should have length 2. {point} given instead."
    if type(point) is np.ndarray:
        assert point.shape == (2,), f"point should have shape (2,). {point} given instead."
    
    i = int(point[1]) # maybe sometimes should be mask.shape[0] - point[1]
    j = int(point[0])
    if i >= mask.shape[0] or i < 0:
        return False
    elif j >= mask.shape[1] or j < 0:
        return False
    else:
        return True
    
def get_perpendicular_growth_at_point(mask0, mask1, contour, contour_ind, 
                        step_size, target, background, num_points
                       ):
    """ For a given point on the comb contour, find the closest comb 
    edge in next mask along line perpendicular to local contour tangent.
    
    Args:
        mask0: comb mask for starting frame
        mask1: comb mask for next frame
        contour: contour in mask0
        step_size: how far along line to look for next intersection
        target: comb value in mask 
        background: background value in mask
        num_points: how many points to average on each size when calculating tangent
    """
    
    contour_val0 = target #point_mask_value(mask0, contour[contour_ind, 0])
    contour_val1 = point_mask_value(mask1, contour[contour_ind, 0])
    
    if contour_val0 == contour_val1:
        # Search away from the target blob within the contour
        # in mask0. Move along line until hit background.
        # (The blob has gotten bigger)
        direction = away_from_self(mask0, contour, contour_ind, target, num_points)
        point, distance = get_target_intersect(mask1, contour, contour_ind, 
                                               step_size, direction, 
                                               target=target, 
                                               num_points=num_points,
                                               anti_target=True
                                              )
    else:
        # Search into the target blob within the contour in mask0.
        # Move along the line until target is hit.
        # (The blob has gotten smaller)
        direction = -away_from_self(mask0, contour, contour_ind, target, num_points)
        point, distance = get_target_intersect(mask1, contour, contour_ind, 
                                               step_size, direction, 
                                               target=target, 
                                               num_points=num_points,
                                               anti_target=False
                                              )
        
    return point, distance

    
    
def get_target_intersect(mask, contour, contour_ind, step_size, 
                         direction, target=1,
                         num_points=10, anti_target=False
                        ):
    """ From specific location on contour, find the point along
        the line perpendicular to the contours local tangent
        where it intersects with a specific value in a mask.
        
        Args:
            mask: 2D array
            contour: opencv contour
            contour_ind: location on contour
            step_size: how far along line to look for next intersection
            direction: Whether to move away from starting point in the positive
                or negative direction along the line (1, or -1)
            target: comb value in mask 
            num_points: how many points to average on each size when calculating tangent
            anti_target: instead looking for target, looks for any value that is not target
            
            
        returns:
            1. location of intersection
            2. length along line at intersection
        
        """   
    distance = 0 # of interesection
    
    dx, dy = get_perpendicular_to_tangent_slope(contour, 
                                                contour_ind, 
                                                num_points=num_points)
    step = np.array([dx, dy]) * step_size * direction
    distance += step_size
    
    starting_point = contour[contour_ind, 0] # contour is shape (n, 1, 2)
    
    point = starting_point + step  
    
    while not is_point_in_target(mask, point, target, anti_target):
        if not is_point_in_mask(mask, point):
            # No intersection before leaving mask
            return None, None
        point += step
        distance += step_size
    # If step size is greater than one, don't know exact intersection point
    # Go backwards to find exact intersection
    small_step = np.array([dx, dy]) * direction
    point -= small_step
    distance -= 1
    while is_point_in_target(mask, point, target, anti_target):
        point -= small_step
        distance -= 1
    
    point += small_step
    distance += 1
    
    return point.astype(int), distance

def away_from_self(mask, contour, contour_ind, target=1, num_points=10):
    """ Return direction along perpendicular line to local contour tanget
    pointing away from blob contour wraps.
    
    Assumes contour is around blob in mask. Target is the value of the blob.
    Also doesn't deal with corner case where contour point is on border of image.
    
    Args:
        mask: blob info
        contour: opencv contour
        contour_ind: location on contour
        target: value of blob in mask
        num_points: how many points to average on each size when calculating tangent
        
    returns -1 or 1
    """
    
    dx, dy = get_perpendicular_to_tangent_slope(contour, 
                                                contour_ind, 
                                                num_points)
    step = np.array([dx, dy]) * 3
    
    start_point = contour[contour_ind, 0]

    if is_point_in_target(mask, start_point+step, target):
        # step in positive direction went into blob
        # so should go negative to get away
        return -1
    else:
        return 1
    
    
    
# def load_nest_images(nest_folder):
#     """ Return list of weeks each with list of frames for that week. 
    
#     Args:
#         nest_file: path to nest folder which has day folders with frame masks
#     """
    
#     nest_images = []
#     week_folders = sorted(glob.glob(os.path.join(nest_folder, "*")))
    
#     for week_folder in week_folders:
#         frame_files = glob.glob(os.path.join(week_folder, "*.JPG"))
#         frame_files = sorted(frame_files)
#         nest_images.append([])
#         for frame_file in frame_files:
#             frame_image = cv2.imread(frame_file)
#             nest_images[-1].append(frame_image)
            
#     return nest_images

# def nest_files(nest_folder):
#     """ Return list of weeks each with list of frame files for that week. 
    
#     Args:
#         nest_file: path to nest folder which has day folders with frame masks
#     """
    
#     nest_files = []
#     week_folders = sorted(glob.glob(os.path.join(nest_folder, "*")))
    
#     for week_folder in week_folders:
#         frame_files = glob.glob(os.path.join(week_folder, "*.JPG"))
#         frame_files = sorted(frame_files)
#         nest_files.append([])
#         for frame_file in frame_files:
#             nest_files[-1].append(frame_file)
            
#     return nest_files