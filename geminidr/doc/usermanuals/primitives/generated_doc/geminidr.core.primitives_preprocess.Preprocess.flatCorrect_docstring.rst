
This primitive will divide each SCI extension of the inputs by those
of the corresponding flat. If the inputs contain VAR or DQ frames,
those will also be updated accordingly due to the division on the data.
If no flatfield is provided, the calibration database(s) will be
queried.

If the flatfield has had a QE correction applied, this information is
copied into the science header to avoid the correction being applied
twice.

Parameters
----------
suffix: str
    suffix to be added to output files
flat: str
    name of flatfield to use
do_cal: str [procmode|force|skip]
    perform flatfield correction?
