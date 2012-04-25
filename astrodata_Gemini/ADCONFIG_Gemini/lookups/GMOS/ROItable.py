# lists the valid ROIs for each ROI name
# The ROIs here are given in physical (ie unbinned) pixels, and are 1- based.
# We provide a list of rois for each name as the definitions get changed once
# once in a while and we don't change the name.

# Format is: { "ROI Name" : [[list of], [rois for], [this ROI name]}
gmosRoiSettings = {
    "Full Frame" : [
        [1, 6144, 1, 4608]
        ],
    "CCD2" :[
        [2049, 4096, 1, 4608]
    ],
    "Central Spectrum" : [
        # This got adjusted by 1 pixel sometime circa 2010
        [1,6144, 1793,2816],
        [1,6144, 1792,2815]
    ],
    "Central Stamp" : [
        [2923, 3222, 2155, 2454]
    ]
}
