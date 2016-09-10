class TRECS_SPECT(DataClassification):
    name = "TRECS_SPECT"
    usage = "Applies to all spectroscopic datasets from the TRECS instrument"
    parent = "TRECS"
    requirement = ISCLASS("TRECS") & NOT(ISCLASS("TRECS_IMAGE"))

newtypes.append(TRECS_SPECT())
