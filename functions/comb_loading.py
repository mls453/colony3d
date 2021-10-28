import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_organized_colony_names(beeframe_meta):
    """ Return names of all colonies that have frame and side info.
    NOTE: will also return nests that are only partially organized
    
    Args:
        beeframe_meta: dataframe with columns like 'img_to_text_d_TOEDIT.csv'
    """
    has_frame_info = ~beeframe_meta['beeframe'].isna()
    has_side_info = ~beeframe_meta['side'].isna()
    colonies = beeframe_meta.loc[has_frame_info&has_side_info, 'colony']
    colonies = colonies.unique()
    return colonies.tolist()

def load_side_mask(masks_folder, day_info, frame_num, side):
    """Load comb mask from assosiated with frame_num and side in day_info.
    
    Args:
        masks_folder: path to nest_photos folder
        day_info: dataframe info like in 'img_to_text_df_TOEDIT.csv' but
            just for one day of one colony. Only rows that have frame 
            and side info.
        frame_num: frame num for mask
        side: frame side of mask
        
    Return:
        2D numpy array of comb mask or None if no file.
    """
    
    frame_rows = day_info['beeframe'] == frame_num
    target_side = day_info['side'] == side
    mask_filename = day_info.loc[frame_rows&target_side, 'filename']
    if len(mask_filename) == 0:
        return None
    if len(mask_filename) > 1:
        colony_name = day_info.iloc[0]['colony']
        date = day_info.iloc[0]['date']
        print(f"Warning multiple images for {colony_name} {date}",
              f" position {frame_num} side {side}.",
              f"Taking first one."
             )
    # Get the actual filename (and the first one if there is overlap)
    mask_filename = mask_filename.iloc[0]
    mask_filename = os.path.splitext(mask_filename)[0]
    mask = cv2.imread(os.path.join(masks_folder, mask_filename+".png"), 
                      cv2.IMREAD_GRAYSCALE
                      )
    return mask


def _combine_ab_mask(side_a, side_b, mirror_b):
    """ Merge comb mask of side a and mask for side b into one comb mask.
    If either mask has comb at a point, will label that point as comb.
    Assumes 0 is background and wood and comb are 1 and 2 in mask.
    
    Args:
        side_a: 2D numpy array, comb mask
        side_b: 2D numpy_array, comb_mask
        mirror_b: should b be horizonattally
            mirrored to match a's orientation
        
    Return:
        2D numpy array
    """
    
    if side_a.shape != side_b.shape:
        raise RuntimeError(f"side_a.shape {side_a.shape} "
                           f"must match side_b.shape {side_b.shape}"
                          )
    if mirror_b:
        aligned_b = side_b[:,::-1]
    else:
        aligned_b = side_b
    
    comb_mask = np.copy(side_a)
    # Add all wood or comb present in b to a
    comb_mask = np.where(aligned_b > 0, aligned_b, side_a)
    # Any wood in b that replaced comb in a should be flipped back 
    comb_mask = np.where(side_a == 2, side_a, comb_mask)
    
    return comb_mask

def load_colony_comb_at_date(colony_df, date, folder_root,
                             masks_folder_name, combine_ab, 
                             mirror_b=False
                            ):
    """ Load colony comb info for date into array.
    Just comb info, so assumes front side and back
    side are the same.
        
    Args:
        colony_df: dataframe info like in 'img_to_text_df_TOEDIT.csv' but
            just for one colony. Only rows that have frame and side info.
        date: date user wants to load. Like: 20210412
        folder_root: path to the "nest_photos" folder
        masks_folder_name: name of the folder the masks that should be loaded are in.
            Like 'warped_masks' for instance.
        combine_ab: if True, mirror comb mask for side b label as comb if either
            a side or b side has comb labeled to account for model error on one
            side but not other. Assumes missed comb in more likely than false 
            comb. If False, just use side a.
        mirror_b: should side b be mirrored if combining a and b side
        
                
    Returns: num_frames x mask_height x mask_width
    """

    colony_name = colony_df.iloc[0]['colony']
    masks_folder = os.path.join(folder_root, colony_name, 
                                str(date), masks_folder_name
                                )
    
    day_rows = colony_df['date'] == date
    day_info = colony_df.loc[day_rows]
    
    nest = []
    for frame_num in range(1, 11):
        side_a = load_side_mask(masks_folder, day_info, frame_num, 'a')
        if side_a is None:
            if combine_ab:
                side_b = load_side_mask(masks_folder, day_info, frame_num, 'b')
                if side_b is None:
                    print(f"No valid info for frame {frame_num},",
                          f"{colony_name}, {date}."
                         )
                    print("returning None")
                    return None
                if mirror_b:
                    side_a = side_b[:,::-1]
                else:
                    side_a = side_b
            else:
                print(f"No valid info for frame {frame_num} a,",
                      f"{colony_name}, {date}."
                     )
                nest.append(None)

        if combine_ab:
            side_b = load_side_mask(masks_folder, day_info, frame_num, 'b')
            if side_b is not None:
                side_a = _combine_ab_mask(side_a, side_b, mirror_b)
        nest.append(side_a)
    
    nest = np.stack(nest, 0)
    return nest

def load_colony_comb(beeframe_meta, colony_name, folder_root, 
                     masks_folder_name, combine_ab, mirror_b=False):
    """ Load colony comb info in 4D array (days x frames x height x width).
    Just comb info, so assumes front side and back
    side are the same.
        
    Args:
        beeframe_meta: dataframe info like in 'img_to_text_df_TOEDIT.csv'
        colony_name: name of colony
        folder_root: path to the "nest_photos" folder
        masks_folder_name: name of the folder the masks that should be loaded are in.
            Like 'warped_masks' for instance.
        combine_ab: if True, mirror comb mask for side b label as comb if either
            a side or b side has comb labeled to account for model error on one
            side but not other. Assumes missed comb in more likely than false 
            comb. If False, just use side a.
        mirror_b: should side b be mirrored if combining a and b side 
        
                
    Returns: days x num_frames x mask_height x mask_width
    """
    
    colony_rows = beeframe_meta['colony'] == colony_name

    has_frame_info = ~beeframe_meta['beeframe'].isna()
    has_side_info = ~beeframe_meta['side'].isna()
    is_organized = has_frame_info & has_side_info

    colony_df = beeframe_meta.loc[colony_rows & is_organized]
    
    colony = []
    
    dates = sorted(colony_df['date'].unique())
    for date in dates:
        colony_day = load_colony_comb_at_date(colony_df, date, 
                                              folder_root, masks_folder_name,
                                              combine_ab, mirror_b
                                             )
        if colony_day is None:
            print(f"Colony is missing frame info. Returning day until this point.")
            break
        colony.append(colony_day)
    if len(colony) >= 1:
        colony = np.stack(colony, axis=0)
    else:
        colony = None
    return colony

def create_colonies_summary(colonies, comb_class, wood_class,
                            num_interior_pixels=None
                           ):
    """Sumarize frame contents for list of colonies.
    
    For each frame compute number of wood and comb pixels.
    And fraction of interior if given.
    
    Args:
        colonies: list of colony arrays (WxFxHxW)
        comb_class: the value of comb in frame masks
        wood_class: the value of wood in frame masks
        num_interior_pixels: how many pixels within the wooden
            border of the frame. Used to normalize pixel counts
            of comb (and wood but less meaningful)
    
    Returns:
        data frame with columns: colony, week, frame, type,
        wood_pixels, comb_pixels, wood_fraction, comb_fraction
    """
    frame_summaries = []
    for colony_dict in colonies:
        colony = colony_dict['colony']
        colony_name = colony_dict['name']
        colony_type = colony_dict['type']
        for week_num, week in enumerate(colony):
            for frame_num, frame in enumerate(week):
                wood_pixel_count = np.sum(frame==wood_class)
                comb_pixel_count = np.sum(frame==comb_class)
                frame_summary = {'colony': colony_name,
                                 'week': week_num,
                                 'frame': frame_num,
                                 'type': colony_type,
                                 'wood_pixels': wood_pixel_count,
                                 'comb_pixels': comb_pixel_count,
                                 'wood_fraction': np.nan,
                                 'comb_fraction': np.nan
                                }
                if num_interior_pixels:
                    frac_wood = wood_pixel_count / num_interior_pixels
                    frac_comb = comb_pixel_count / num_interior_pixels
                    frame_summary['wood_fraction'] = frac_wood
                    frame_summary['comb_fraction'] = frac_comb
                frame_summaries.append(frame_summary)
    frame_contents = pd.DataFrame(frame_summaries)
    return frame_contents

def visualize_colony(colony, figsize=(20, 10), max_class=2,
                     colony_name=None):
    """ Show all frames in colony in grid.
    
    Args: 
        colony: (WxFxHxW) array
        figsize: size to draw figure
        max_class: number of classes to display in frames
        colony_name: name of the colony
        
    Returns:
        figure (plots figure)
    """
    num_weeks = colony.shape[0]
    num_frames = colony.shape[1]
    fig, axs = plt.subplots(num_weeks, num_frames, figsize=figsize)
    for week, week_frames in enumerate(colony):
        for frame_ind, frame in enumerate(week_frames):
            axs[week, frame_ind].imshow(frame, vmin=0, vmax=max_class)
            axs[week, frame_ind].get_xaxis().set_visible(False)
            axs[week, frame_ind].get_yaxis().set_visible(False)
            axs[week, frame_ind].set_ylabel(f"week {week}")
            axs[week, frame_ind].set_xlabel(f"frame {frame_ind}")

    if colony_name:
        _ = fig.suptitle(f"Colony {colony_name}")
        
    return fig

