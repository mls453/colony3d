import matplotlib.pyplot as plt
   
from mask_processing import get_distance_to_class

def visualize_get_distance_to_class(point, mask, class_ids, colors=None):
    """ Display point and closest point that is in class_id.
        Uses function 'get_distance_to_class'
        
        Args:
            point: (i, j) numpy indexing (from top left: (row, column))
            mask: 2D numpy array
            class_ids: list of values of class of interest in mask
            colors: list of plt colors, first is point, should be as
                long as class_ids + 1
    
        Return:
            plot image with dots
        
    """
    if colors is None:
        # When colors isn't specified, plot point in red
        # and all other points in green
        colors = ['r']
        for _ in class_ids:
            colors.append('g')
    if len(colors) != len(class_ids) + 1:
        raise RuntimeError(f"Colors must be length {len(class_ids)+1}",
                           f"but is actually length {len(colors)}.")
    plt.imshow(mask)
    plt.scatter(*point[::-1], c=colors[0])
    for class_ind, class_id in enumerate(class_ids):
        distance, comb_position = get_distance_to_class(point, mask, class_id)
        if distance:
            print(f"distance to {class_id}: {distance}")
            plt.scatter(*comb_position[::-1], c=colors[class_ind+1])

                                                    