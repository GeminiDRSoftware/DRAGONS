class TRECS_RAW(DataClassification):
    editprotect=True
    name="TRECS_RAW"
    usage = 'Un-"prepared" TRECS data.'
    typeReqs= ['TRECS']
    phuReqs= {'{prohibit}TPREPARE': ".*?" }
    
newtypes.append(TRECS_RAW())
