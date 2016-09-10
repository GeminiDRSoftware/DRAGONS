
class GNIRS_IMAGE(DataClassification):
    name="GNIRS_IMAGE"
    usage = "Applies to any IMAGE dataset from the GNIRS instrument."
    parent = "GNIRS"
    requirement = AND([ ISCLASS('GNIRS'),
                        PHU(ACQMIR='In'),
                        NOT(ISCLASS("GNIRS_DARK"))
                      ])

newtypes.append(GNIRS_IMAGE())
