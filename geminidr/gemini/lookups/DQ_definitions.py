import numpy as np
datatype = np.uint16

bad_pixel = 1
non_linear = 2
saturated = 4
cosmic_ray = 8
no_data = 16
overlap = 32
unilluminated = 64

fail = bad_pixel | saturated | cosmic_ray | no_data
