
class TRECS(DataClassification):
    name="TRECS"
    usage = "Applies to all datasets from the TRECS instrument."
    parent = "GEMINI"
    requirement = PHU(INSTRUME='TReCS')

newtypes.append(TRECS())

class TRECS_IMAGE(DataClassification):
    name="TRECS_IMAGE"
    usage = "Applies to all IMAGE datasets from the TRECS instrument."
    parent = "TRECS"
    requirement = ISCLASS("TRECS") & PHU({"{prohibit}GRATING":".*?[mM]irror.*?"})

newtypes.append(TRECS_IMAGE())

class TRECS_SPECT(DataClassification):
    name="TRECS_SPECT"
    usage = "Applies to all SPECTral datasets from the TRECS instrument."
    parent = "TRECS"
    requirement = ISCLASS("TRECS") & NOT(ISCLASS("TRECS_IMAGE"))

newtypes.append(TRECS_SPECT())
