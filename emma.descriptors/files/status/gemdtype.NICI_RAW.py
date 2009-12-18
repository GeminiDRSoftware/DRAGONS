class NICI_RAW(DataClassification):
    editprotect=True
    name="NICI_RAW"
    usage = 'Un-"prepared" NICI data.'
    typeReqs= ['NICI']
    phuReqs= {'{prohibit}NCPREPARE': ".*?" }
    
newtypes.append(NICI_RAW())
