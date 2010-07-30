
class GMOS_IMAGE(DataClassification):
    name="GMOS_IMAGE"
    usage = ""
    parent = "GMOS"
    requirement = AND( ISCLASS(      "GMOS"),
                       PHU(GRATING="MIRROR") )

newtypes.append(GMOS_IMAGE())
