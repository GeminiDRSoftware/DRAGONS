class RAW(DataClassification):
    editprotect=True
    name="UNPREPARED"
    usage = 'Un-"prepared" data.'
    phuReqs= {'{prohibit}GEM-TLM': ".*?" }
    
newtypes.append(RAW())
