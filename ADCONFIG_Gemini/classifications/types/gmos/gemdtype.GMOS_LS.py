
class GMOS_LONGSLIT(DataClassification):
    name="GMOS_LONGSLIT"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {  'OBSMODE': 'LONGSLIT',
                'OBSTYPE': 'OBJECT'
            }

newtypes.append(GMOS_LONGSLIT())
