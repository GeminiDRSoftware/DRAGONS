
This darkCorrect primitive will subtract the signal of the processed
dark from the signal of the input frames.  The variance and data
quality mask will be updated as appropriate, if they exist.  If no
dark is provided, the calibration database(s) will be queried.

Parameters
----------
suffix: str
    Suffix to be added to output files
dark: str/list
    Filenname of the dark(s) to subtract. If no filename is provided,
    the calibration databases will be queried to find a matching
    processed dark for each of the input datasets.  If one dark
    filename is provided, it will be used on all the input frames.  If
    more than one dark filename is provided, the number of darks must
    match the number of input frames.
do_dark: str
    Require the dark subtraction?  If set to `procmode`. whether the
    dark subtraction is required or not depends on the processing mode:
    it is required for 'sq' mode but optional for 'ql' and 'qa'
    modes.  If set to `force`, the dark subtraction is required for all
    processing modes.  If set to `skip`, the dark subtraction is
    skipped, the primitive will not attempt to find a dark frame, and
    no changes will be made to the input frames.
