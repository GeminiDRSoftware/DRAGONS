import numpy as np

datatype = np.uint16
max = datatype(np.iinfo(datatype).max)

good = datatype(0)
bad_pixel = datatype(1)
non_linear = datatype(2)
saturated = datatype(4)
cosmic_ray = datatype(8)
no_data = datatype(16)
overlap = datatype(32)
unilluminated = datatype(64)

fail = bad_pixel | saturated | cosmic_ray | no_data
not_signal = max ^ (non_linear | saturated)
