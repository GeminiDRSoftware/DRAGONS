class MICHELLE_RAW(DataClassification):
    editprotect=True
    name="MICHELLE_RAW"
    usage = 'Un-"prepared" MICHELLE data.'
    typeReqs= ['MICHELLE']
    phuReqs= {'{prohibit}MPREPARE': ".*?" }
    
newtypes.append(MICHELLE_RAW())
