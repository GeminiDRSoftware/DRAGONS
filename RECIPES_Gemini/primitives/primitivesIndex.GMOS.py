# this is added to the reduction object dictionary, but only one
# reduction object per astro data type.
# NOTE: primitives are the member functions of a Reduction Object.

localPrimitiveIndex = {
    "GEMINI": ("reduction_GEMINI.py", "GEMINIReduction"),
    "GMOS_OBJECT_RAW": ("reduction_GMOS_OBJECT_RAW.py", "GMOS_IMAGEReduction")
    }
