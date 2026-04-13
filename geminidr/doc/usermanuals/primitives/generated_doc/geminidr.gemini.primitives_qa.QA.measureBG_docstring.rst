
This primitive measures the sky background level for an image by
sampling the non-object unflagged pixels in each extension.

The count levels are then converted to a flux using the nominal
(*not* measured) Zeropoint values - the point being you want to measure
the actual background level, not the flux incident on the top of the
cloud layer necessary to produce that flux level.

Parameters
----------
suffix : str
    suffix to be added to output files
remove_bias : bool
    remove the bias level (if present) before measuring background?
separate_ext : bool
    report one value per extension, instead of a global value?
