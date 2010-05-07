
class GNIRS_SPECT(DataClassification):
    name="GNIRS_SPECT"
    usage = "Applies to any SPECT dataset from the GNIRS instrument."
    parent = "GNIRS"
    requirement = ISCLASS('GNIRS') & PHU(ACQMIR='Out')

newtypes.append(GNIRS_SPECT())
