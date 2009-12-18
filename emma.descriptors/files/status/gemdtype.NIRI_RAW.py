class NIRI_RAW(DataClassification):
    editprotect=True
    name="NIRI_RAW"
    usage = 'Un-"prepared" NIRI data.'
    typeReqs= ['NIRI']
    phuReqs= {'{prohibit}NPREPARE': ".*?" }
    
newtypes.append(NIRI_RAW())
