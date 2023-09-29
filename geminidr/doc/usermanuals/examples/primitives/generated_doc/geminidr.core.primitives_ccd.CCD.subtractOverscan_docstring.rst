
This primitive subtracts the overscan level from the image. The
level for each row (currently the primitive requires that the overscan
region be a vertical strip) is determined in one of the following
ways, according to the *function* and *order* parameters:

:"poly":   a polynomial of degree *order* (1=linear, etc)
:"spline": using *order* equally-sized cubic spline pieces or, if
          order=None or 0, a spline that provides a reduced chi^2=1
:"none":   no function is fit, and the value for each row is determined
          by the overscan pixels in that row

The fitting is done iteratively but, in the first instance, a running
median of the rows is calculated and rows that deviate from this median
are rejected (and used in place of the actual value if function="none")

Parameters
----------
suffix: str
    suffix to be added to output files
niterate: int
    number of rejection iterations
high_reject: float/None
    number of standard deviations above which to reject high pixels
low_reject: float/None
    number of standard deviations above which to reject low pixels
nbiascontam: int/None
    number of columns adjacent to the illuminated region to reject
function: str/None
    function to fit ("chebyshev" | "spline" | "none")
order: int/None
    order of polynomial fit or number of spline pieces
bias_type: str
    For multiple overscan regions, selects which one to use
