
This primitive will subtract each SCI extension of the inputs by those
of the corresponding dark. If the inputs contain VAR or DQ frames,
those will also be updated accordingly due to the subtraction on the
data. If no dark is provided, the calibration database(s) will be
queried.

Parameters
----------
suffix: str
    suffix to be added to output files
dark: str/list
    name(s) of the dark file(s) to be subtracted
do_dark: bool
    perform dark correction?
