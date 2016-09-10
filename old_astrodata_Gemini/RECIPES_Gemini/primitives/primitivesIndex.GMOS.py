# This is added to the reduction object dictionary, but only one reduction
# object per AstroData Type. NOTE: primitives are the member functions of a
# Reduction Object.

localPrimitiveIndex = {
    "GMOS": ("primitives_GMOS.py", "GMOSPrimitives"),
    "GMOS_IMAGE": ("primitives_GMOS_IMAGE.py", "GMOS_IMAGEPrimitives"),
    "GMOS_SPECT": ("primitives_GMOS_SPECT.py", "GMOS_SPECTPrimitives"),
    "GMOS_NODANDSHUFFLE": ("primitives_GMOS_NODANDSHUFFLE.py", "GMOS_NODANDSHUFFLEPrimitives")
    }
