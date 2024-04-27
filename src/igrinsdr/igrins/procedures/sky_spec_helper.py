import numpy as np


def _get_slices(n_slice_one_direction):
    """
    given number of slices per direction, return slices for the
    center, up and down positions.
    """
    n_slice = n_slice_one_direction*2 + 1
    i_center = n_slice_one_direction
    slit_slice = np.linspace(0., 1., n_slice+1)

    slice_center = (slit_slice[i_center], slit_slice[i_center+1])

    slice_up = [(slit_slice[i_center+i], slit_slice[i_center+i+1])
                for i in range(1, n_slice_one_direction+1)]

    slice_down = [(slit_slice[i_center-i-1], slit_slice[i_center-i])
                  for i in range(n_slice_one_direction)]

    return slice_center, slice_up, slice_down


