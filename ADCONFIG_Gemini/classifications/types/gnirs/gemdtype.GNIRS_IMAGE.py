
class GNIRS_IMAGE(DataClassification):
    name="GNIRS_IMAGE"
    usage = "Applies to any IMAGE dataset from the GNIRS instrument."
    parent = "GNIRS"
    requirement = ISCLASS('GNIRS') & PHU(ACQMIR='In')

newtypes.append(GNIRS_IMAGE())
