
This primitive scales input images to have the same effective exposure
time. This can either be provided as a parameter, or the images will be
scaled to match the exposure time of the first image in the input list.

Parameters
----------
suffix: str/None
    suffix to be added to output files
time: float/None
    exposure time to scale to (None => use first image's exposure time)
