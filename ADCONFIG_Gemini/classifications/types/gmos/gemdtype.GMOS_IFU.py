
class GMOS_IFU(DataClassification):
    name="GMOS_IFU"
    usage = "Data taken in the IFU instrument mode with either GMOS instrument"
    typeReqs= ['GMOS']
    phuReqs= {'OBSMODE': 'IFU',
                'OBSTYPE': 'OBJECT'
            }

newtypes.append(GMOS_IFU())
