# this is added to the reduction object dictionary, but only one
# reduction object per astro data type.
# NOTE: primitives are the member functions of a Reduction Object.

localPrimitiveIndex = {
    "GEMINI": ("primitives_GEMINI.py", "GEMINIPrimitives"),
    "GMOS_OBJECT_RAW": ("primitives_GMOS_OBJECT_RAW.py", "GMOS_IMAGEPrimitives")
    }
