class GNIRS_RAW(DataClassification):
    editprotect=True
    name="GNIRS_RAW"
    usage = 'Un-"prepared" GNIRS data.'
    typeReqs= ['GNIRS']
    phuReqs= {'{prohibit}NSPREPARE': ".*?" }
    
newtypes.append(GNIRS_RAW())
