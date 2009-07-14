class GMOS_LS(DataClassification):
    name="GMOS_LS"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {  'OBSMODE': 'LONGSLIT',
                'OBSTYPE': 'OBJECT'
            }

newtypes.append(GMOS_LS())
