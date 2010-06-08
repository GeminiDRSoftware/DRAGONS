
class GMOS_IMAGE(DataClassification):
    name="GMOS_IMAGE"
    usage = ""
    parent = "GMOS"
    requirement = ISCLASS("GMOS") & PHU(GRATING="MIRROR")

newtypes.append(GMOS_IMAGE())
