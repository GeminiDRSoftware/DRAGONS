
class GMOS_IMAGE(DataClassification):
    name="GMOS_IMAGE"
    usage = ""
    parent = "GMOS"
    requirement = AND(ISCLASS("GMOS"),
                      PHU(GRATING="MIRROR"),
                      NOT(ISCLASS("GMOS_BIAS")))

newtypes.append(GMOS_IMAGE())
