
This flatCorrect primitive will divide the signal of the inputs the
signal of the processed flat. The variance and data quality mask will
be updated as appropriate, if they exist. If no flatfield is provided,
the calibration database(s) will be queried.

If the flatfield has had a QE correction applied, this information is
copied into the science header to avoid the correction being applied
twice.

Parameters
----------
suffix: str
    Suffix to be added to output files
flat: str/list of str
    Filename of the flatfield(s) to use. If no filename is provided,
    the calibration databases will be queried to find a matching
    processed flat for each of the input datasets.  If one flat
    filename is provided, it will be used on all the input frames.  If
    more than one flat filename is provided, the number of flats must
    match the number of input frames.
do_cal: str
    Require the flatfielding?  If set to `procmode`. whether the
    flat correction is required or not depends on the processing mode:
    it is required for 'sq' mode but optional for 'ql' and 'qa'
    modes.  If set to `force`, the flat correction is required for all
    processing modes.  If set to `skip`, the flat correction is
    skipped, the primitive will not attempt to find a flat frame, and
    no changes will be made to the input frames.
