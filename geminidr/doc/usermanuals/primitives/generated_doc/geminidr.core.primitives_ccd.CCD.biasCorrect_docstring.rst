
The biasCorrect primitive will subtract the science extension of the
input bias frames from the science extension of the input science
frames. The variance and data quality extension will be updated, if
they exist. If no bias is provided, the calibration database(s) will
be queried.

Parameters
----------
suffix: str
    suffix to be added to output files
bias: str/list of str
    bias(es) to subtract
do_cal: str
    perform bias subtraction?
