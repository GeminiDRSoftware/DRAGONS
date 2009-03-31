class GMOS_RAW(DataClassification):
    editprotect=True
    name="GMOS_RAW"
    usage = 'Un-"prepared" GMOS data.'
    typeReqs= ['GMOS']
    phuReqs= {'{prohibit}GPREPARE': ".*?" }
    
newtypes.append(GMOS_RAW())
