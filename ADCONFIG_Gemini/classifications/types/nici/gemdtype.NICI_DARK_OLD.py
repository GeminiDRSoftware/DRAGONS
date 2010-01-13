
class NICI_DARK_OLD(DataClassification):
    name="NICI_DARK_OLD"
    usage = ""
    typeReqs= ['NICI']
    phuReqs= {'OBSTYPE': 'FLAT',
              'GCALSHUT': 'CLOSED',
  	     }

newtypes.append(NICI_DARK_OLD())
