import numpy as np
from functools import wraps

def history_decorator(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Create the formatted string with the method name and arguments
        method_name = func.__name__
        arg_strs = [f"{arg_name}={repr(arg_value)}" for arg_name, arg_value in zip(func.__code__.co_varnames[1:], args)]
        kwarg_strs = [f"{key}={repr(value)}" for key, value in kwargs.items()]
        history_str = f"{method_name}__{'__'.join(arg_strs + kwarg_strs)}"

        # Append the formatted string to self.history
        
        
        # Call the original method
        res = func(self, *args, **kwargs)
        self.history.append(history_str)
        return res
    
    return wrapper

def crop(img, coordinates, buffer):
    # crop the img to a minimal square that covers the coordinates + buffer region around each size
    # returns cropped img, and coordinates wrt img
    coordinates_ = coordinates.copy().astype("int")
    min_0 = max(np.min(coordinates_[:,0]) - buffer,0)
    max_0 = min(np.max(coordinates_[:,0]) + buffer, img.shape[0])
    min_1 = max(np.min(coordinates_[:,1]) - buffer,0)
    max_1 = min(np.max(coordinates_[:,1]) + buffer, img.shape[1])

    cropped_img = img.copy()
    #cropped_img = cropped_img[min_0: max_0 + 1, min_1: max_1 + 1] #This is correct. 
    cropped_img = cropped_img[min_0: max_0, min_1: max_1] #-> this is wrong, this is what was in the original bin2cell
    #To have exactly the same stardist output, uncomment the above...

    new_coordinates = coordinates.copy()
    new_coordinates[:,0] -= min_0
    new_coordinates[:,1] -= min_1

    return cropped_img, new_coordinates

def is_rgb(img):
    if img.ndim == 2:
        return False
    elif img.ndim == 3:
        return True
    else:
        raise ValueError("I did not plan this")
    
def imshow(img, ax):
    rgb = is_rgb(img)
    if not(rgb):
        ax.imshow(img, cmap = "gray", origin = "upper", vmin = 0, vmax = 255)
    elif rgb:
        ax.imshow(img, origin = "upper")
    return ax