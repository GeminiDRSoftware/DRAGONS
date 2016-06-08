
class GNIRS_SPECT(DataClassification):
    name="GNIRS_SPECT"
    usage = "Applies to any SPECT dataset from the GNIRS instrument."
    parent = "GNIRS"
    requirement = AND([  ISCLASS('GNIRS'),
                         PHU(ACQMIR='Out'),
                         NOT(ISCLASS("GNIRS_DARK"))
                      ])   

newtypes.append(GNIRS_SPECT())
