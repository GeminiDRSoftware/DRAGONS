
class GMOS_MOS(DataClassification):
    name="GMOS_MOS"
    usage = ""
    parent = "GMOS_SPECT"
    requirement = ISCLASS('GMOS') & PHU(OBSMODE='MOS',      
                                        OBSTYPE='OBJECT')

newtypes.append(GMOS_MOS())
