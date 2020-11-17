# cython: language_level=3
cimport cython

def landstat(double [:] landscape, int [:] coords, int [:] len_axes,
             int num_axes, int num_coords):
    cdef int c, coord, i, j, l, ok
    cdef float sum=0.
    cdef int sum2=0

    for i in range(num_coords):
        c = i
        ok = 1
        l = 0
        for j in range(num_axes-1, -1, -1):
            coord = coords[c]
            if coord >=0 and coord < len_axes[j]:
                if j < num_axes - 1:
                    l += coord * len_axes[j+1]
                else:
                    l += coord
            else:
                ok = 0
            c += num_coords
        if ok:
            sum += landscape[l]

    return sum
