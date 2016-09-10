# lists the valid ROIs for each ROI name
# The ROIs here are given in physical (ie unbinned) pixels, and are 1- based.
# We provide a list of rois for each name as the definitions get changed once
# once in a while and we don't change the name.

##M The addition of the Hamamatsu values may actually be a bit of a hack

# Format is: { "ROI Name" : [[list of], [rois for], [this ROI name]}
gmosRoiSettings = {
    "Full Frame" : [
        # EEV / e2vDD CCDs
        [1, 6144, 1, 4608],
        # Hamamatsu GMOS-S
        [1, 6144, 1, 4224]
        ],
    "CCD2" :[
        # EEV / e2vDD CCDs
        [2049, 4096, 1, 4608],
        # Hamamatsu GMOS-S
        [2049, 4096, 1, 4224]
    ],
    "Central Spectrum" : [
        # This got adjusted by 1 pixel sometime circa 2010
        [1, 6144, 1793, 2816],
        [1, 6144, 1792, 2815],
        # Hamamatsu GMOS-S
        [1, 6144, 1625, 2648]
    ],
    "Central Stamp" : [
        # EEV and e2vDD CCDs
        [2923, 3222, 2155, 2454],
        # GMOS-S Hamamatsu CCDs
        [2923, 3222, 1987, 1286]
    ]
}
