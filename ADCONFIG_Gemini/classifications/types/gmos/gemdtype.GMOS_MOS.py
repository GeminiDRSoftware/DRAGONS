
class GMOS_MOS(DataClassification):
    name="GMOS_MOS"
    usage = ""
    requirement = ISCLASS('GMOS') & PHU(OBSMODE='MOS',      
                                        OBSTYPE='OBJECT')

newtypes.append(GMOS_MOS())
