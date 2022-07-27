""" A range of functions commonly used across tasks related to the comb segmentation data."""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_comb_types():
    comb_classes = ["background",
                    "wood",
                    "worker",
                    "drone",
                    "queen",
                    "bee"
                   ]
    return comb_classes


def get_content_types():
    content_classes = ["background",
                       "wood",
                       "comb",
                       "pollen",
                       "nectar",
                       "brood",
                       "eggs",
                       "capped-honey",
                       "capped-brood",
                       "queen-cup",
                       "queen-cell",
                       "bees"
                      ]
    return content_classes

def get_combined_classes():
    orig_comb_classes = ["background",
                         "wood",
                         "comb",
                         "pollen",
                         "nectar",
                         "brood",
                         "eggs",
                         "capped-honey",
                         "capped-brood",
                         "eggs-drone",
                         "pollen-drone",
                         "comb-drone",
                         "queen-cup",
                         "nectar-drone",
                         "capped-brood-drone",
                         "brood-drone",
                         "queen-cell",
                         "bees"
                        ]
    return orig_comb_classes

def default_frame_array_size():
    """ Returns the standard output array size of comb labels from model."""
    return (3200, 4960)

def get_frame_filename(frame_positions, frame_num, side):
    """get filename based on frame_num and side in frame_positions.
    
    Args:
        frame_positions: dataframe info like in 'img_to_text_df_TOEDIT.csv' but
            just for one colony at single date. Only rows that have frame 
            and side info.
        frame_num: frame num for mask
        side: frame side of mask
        
    Return:
        Corresponding filename in the 'filename' column of frame_positions.
        With file extension removed.
    """
    
    frame_rows = frame_positions['beeframe'] == frame_num
    target_side = frame_positions['side'] == side
    filenames = frame_positions.loc[frame_rows&target_side, 'filename']
    if len(filenames) == 0:
        return None
    if len(filenames) > 1:
        colony_name = frame_positions.iloc[0]['colony']
        date = frame_positions.iloc[0]['date']
        print(f"Warning multiple images for {colony_name} {date}",
              f" position {frame_num} side {side}.",
              f"Taking first one."
             )
    # Get the actual filename (and the first one if there is overlap)
    filename = filenames.iloc[0]
    filename = os.path.splitext(filename)[0]

    return filename

def load_colony(colony_folder, label_type, dates, colony_frame_positions, 
               downsample=None, verbose=True):
    """Load colony data of given label type at specified dates.
    
    Args:
        colony_folder: path folder containing colony data in date subfolders 
        label_type: name of folder that contains the .npy files with the relevant
            comb information.
        dates: list of dates
        colony_frame_positions: dataframe info like in 'img_to_text_df_TOEDIT.csv' 
            but just for one colony.
            and side info.
        downsample: load data arrays with shape / downsample 
        verbose: if True print info about missing frame data
        
    Return:
        Dict with dates as keys and 20 x h x w arrays as values
            """

    num_frames = 10
    side_names = ["a", "b"]
    
    colony = {}

    for date in dates:
        date_rows = colony_frame_positions['date'] == date
        date_frame_positions = colony_frame_positions[date_rows]
        date_folder = os.path.join(colony_folder, date, label_type)
        framesides = []
        for frame_num in range(1, num_frames+1): # frames labeled 1 through 10
            for side_name in side_names:
                filename = get_frame_filename(date_frame_positions, frame_num, 
                                              side_name)
                if filename is None:
                    if verbose:
                        print(f"{date}: frame {frame_num}, side {side_name} wasn't found.")
                    frameside = np.ones(default_frame_array_size(), dtype=np.uint8) * 255
                else:
                    frameside_file = os.path.join(date_folder, f"{filename}.npy")
                    frameside = np.load(frameside_file)
                if downsample:
                    scale = 1 / downsample
                    frameside = cv2.resize(frameside, (0,0), fx=scale, fy=scale,
                                           interpolation=cv2.INTER_NEAREST) 
                framesides.append(frameside)
        colony[date] = np.stack(framesides)
    
    return colony

def show_colony(colony, num_classes=None, title=None):
    """ Make figure with all frame sides of colony.
            Top row is side a and bottom row is side b with frame 1 on the left
        
        Args:
            colony: 20 x h x w array with storing frame info
            num_classes: max number of classes in the colony data
            title: title for the plot
    """
    num_rows = 2
    num_columns = 10
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 3))
    for column in range(10):
        for row in range(2):
            axs[row, column].imshow(colony[num_rows*column + row], 
                                    vmin=0, vmax=num_classes, 
                                    interpolation="nearest")
            axs[row, column].axis('off')
    fig.suptitle(title)
    
def create_class_count_df(colony, class_names):
    """ Create a dataframe with date and all class name columns and coresponding counts.
    
    Args:
        colony: Dict with dates as keys and 20 x h x w arrays as values.
        class_names: list of class names where the index corresponds with integer
            value for that class in the colony arrays.
            
    Returns a data frame.
    """
    
    date_counts = []
    for date, colony_data in colony.items():
        counts = np.bincount(np.ravel(colony_data), minlength=len(class_names))
        date_counts.append({"date": date})
        for class_name, count in zip(class_names, counts):
            date_counts[-1][class_name] = count
    date_counts = pd.DataFrame(date_counts)
    return date_counts

def get_dates(colony_folder):
    """ Get list of date folders in colony_folder that can be represented as numbers.
    
    Args:
        colony_folder: folder containing possible date folders
    
    Return list of valid dates (as strings)
    """
    folders = os.listdir(colony_folder)
    valid_dates = []
    for folder in folders:
        try:
            int(folder)
        except ValueError:
            continue
        valid_dates.append(int(folder))
    # Sort as integers
    valid_dates.sort()
    # Return as strings
    valid_dates = [str(d) for d in valid_dates]
    return sorted(valid_dates)

def get_colony_names(root_folder):
    """Get only valid colony folders stored in root folder (SH, CC, DD)."""
    
    folders = os.listdir(root_folder)
    
    colony_names = []
    
    for folder in folders:
        if "CC" in folder:
            colony_names.append(folder)
        elif "DD" in folder:
            colony_names.append(folder)
        elif "SH" in folder:
            colony_names.append(folder)
    return sorted(colony_names)