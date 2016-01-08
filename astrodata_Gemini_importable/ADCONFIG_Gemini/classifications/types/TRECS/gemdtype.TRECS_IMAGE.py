class TRECS_IMAGE(DataClassification):
    name = "TRECS_IMAGE"
    usage = "Applies to all imaging datasets from the TRECS instrument"
    parent = "TRECS"
    requirement = ISCLASS("TRECS") & PHU(GRATING="(.*?)[mM]irror")

newtypes.append(TRECS_IMAGE())
