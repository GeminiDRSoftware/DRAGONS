class RAW(DataClassification):
    editprotect=True
    name="RAW"
    usage = 'Un-"prepared" data.'
    phuReqs= {'{prohibit}GEM-TLM': ".*?" }
    
newtypes.append(RAW())
