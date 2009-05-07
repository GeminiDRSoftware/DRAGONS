
class GMOS_ARC(DataClassification):
    name="GMOS_ARC"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {
                'OBSTYPE': 'ARC'
            }

newtypes.append(GMOS_ARC())
