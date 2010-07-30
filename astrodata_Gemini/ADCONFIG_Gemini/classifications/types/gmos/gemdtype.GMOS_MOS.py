
class GMOS_MOS(DataClassification):
    name="GMOS_MOS"
    usage = ""
    parent = "GMOS_SPECT"
    requirement = OR(AND(ISCLASS('GMOS'),
                         PHU(OBSMODE='MOS',OBSTYPE='OBJECT')),
                     ISCLASS('GMOS_MOS_FLAT'),
                     ISCLASS('GMOS_MOS_TWFLAT'),
                     ISCLASS('GMOS_MOS_ARC'))

                    
                                        
                                        

newtypes.append(GMOS_MOS())
