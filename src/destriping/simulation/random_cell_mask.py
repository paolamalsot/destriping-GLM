import numpy as np
from skimage.draw._random_shapes import warn, SHAPE_GENERATORS, SHAPE_CHOICES

def random_shapes(
    image_shape,
    max_shapes,
    min_shapes=1,
    min_size=2,
    max_size=None,
    shape=None,
    allow_overlap=False,
    num_trials=100,
    rng=None
):
    # Very much inspired from skimage.draw.random_shapes, but instead of assigning random color, do a clear label !
    
    if min_size > image_shape[0] or min_size > image_shape[1]:
        raise ValueError('Minimum dimension must be less than ncols and nrows')
    max_size = max_size or max(image_shape[0], image_shape[1])

    rng = np.random.default_rng(rng)
    user_shape = shape
    image_shape = (image_shape[0], image_shape[1])
    image = np.full(image_shape, -1, dtype=np.int64) #-1 means unassigned
    filled = np.zeros(image_shape, dtype=bool)
    labels = []

    num_shapes = rng.integers(min_shapes, max_shapes + 1)
    shape = (min_size, max_size)
    shape_idx = 0
    for shape_idx_try in range(num_shapes):
        if user_shape is None:
            shape_generator = rng.choice(SHAPE_CHOICES)
        else:
            shape_generator = SHAPE_GENERATORS[user_shape]
        for _ in range(num_trials):
            # Pick start coordinates.
            column = rng.integers(max(1, image_shape[1] - min_size))
            row = rng.integers(max(1, image_shape[0] - min_size))
            point = (row, column)
            try:
                indices, label = shape_generator(point, image_shape, shape, rng)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                indices = []
                continue
            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or not filled[indices].any():
                image[indices] = shape_idx
                shape_idx += 1
                filled[indices] = True
                break
        else:
            warn(
                'Could not fit any shapes to image, '
                'consider reducing the minimum dimension'
            )

    return image, labels