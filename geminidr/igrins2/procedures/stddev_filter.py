
# standard deviation filter
from scipy.ndimage.filters import uniform_filter
def window_stdev(arr, radius):
    c1 = uniform_filter(arr, radius*2, mode='constant',
                        #origin=-radius
                        )
    c2 = uniform_filter(arr*arr, radius*2, mode='constant',
                        #origin=-radius
                        )
    r = ((c2 - c1*c1)**.5) #[:-radius*2+1]
    return r
