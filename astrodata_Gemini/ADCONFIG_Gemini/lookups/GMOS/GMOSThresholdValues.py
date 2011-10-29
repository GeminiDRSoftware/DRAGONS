# Dictionary of various threshold levels used for each type of GMOS Detector
# All values in ADU
# Keyed by purpose of value and DETTYPE
gmosThresholds = {
    # EEV CCDs
    ("saturation", "SDSU II CCD") : 65530,
    ("display",    "SDSU II CCD") : 58000,
    ("processing", "SDSU II CCD") : 45000,
    # e2v CCDs
    ("saturation", "SDSU II e2v DD CCD42-90") : 65530,
    ("display",    "SDSU II e2v DD CCD42-90") : 58000,
    ("processing", "SDSU II e2v DD CCD42-90") : 45000,
    # Hamamatsu CCDs
    ("saturation", "S10892-01") : 65530,
    ("display",    "S10892-01") : 58000,
    ("processing", "S10892-01") : 45000,
}
