
The biasCorrect primitive will subtract the signal of the
processed bias from the signal of the input frames. The variance and
data quality mask will be updated as appropriate, if they exist. If no
bias is provided, the calibration database(s) will be queried.

Each astrodata extension is processed independently, so the bias must
have the same number of extensions as the input frames.

Parameters
----------
suffix: str
    Suffix to be added to output files
bias: str/list of str
    Filename of the bias(es) to subtract. If no filename is provided,
    the calibration databases will be queried to find a matching
    processed bias for each of the input datasets.  If one bias
    filename is provided, it will be used on all the input frames.  If
    more than one bias filename is provided, the number of biases must
    match the number of input frames.
do_cal: str
    Require the bias subtraction?  If set to `procmode`. whether the
    bias subtraction is required or not depends on the processing mode:
    it is required for 'sq' mode but optional for 'ql' and 'qa'
    modes.  If set to `force`, the bias subtraction is required for all
    processing modes.  If set to `skip`, the bias subtraction is
    skipped, the primitive will not attempt to find a bias frame, and
    no changes will be made to the input frames.
