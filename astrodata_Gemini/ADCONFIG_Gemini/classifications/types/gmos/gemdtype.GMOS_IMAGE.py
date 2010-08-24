
class GMOS_IMAGE(DataClassification):
    name="GMOS_IMAGE"
    usage = ""
    parent = "GMOS"
    requirement = OR(
                        AND( ISCLASS(      "GMOS"),
                             PHU(GRATING="MIRROR")),
                        ISCLASS("GMOS_BIAS"))
    

newtypes.append(GMOS_IMAGE())
