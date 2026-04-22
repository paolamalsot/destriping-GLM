# spatialAdata class

Goal: have a class which represents a spatial transcriptomics dataset
It can contain multiple images (h&E), multiple coordinates basis (called coordinates_id)

VIP: we crafted the download_visium_hd such that all 2D coordinates have their first element represent the position on the vertical axis (up-to-down wrt image), and second element left-to-right wrt image.

The images are oriented similarly (but can be slightly tilted/reshaped). They are represented via 2D arrays (or 2D x n_channels) that are to be displayed as if we printed a numpy array.

In regard to the img_df and coordinate_df. Note that there are two scale factors.
The one in coordinate_df relates a coordinate to an image, such that the coordinate tuple must be multiplied by the scalefactor to get the pixel position on the image.
The one in image_df characterizes an image. It describes what is the scalefactor wrt to the full-resolution image. This is useful to scale the dots/bins appropriately.

The "tilt" is encoded in the coordinates themselves.