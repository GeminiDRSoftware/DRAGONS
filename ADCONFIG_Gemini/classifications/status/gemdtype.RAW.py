class RAW(DataClassification):
    editprotect=True
    name="RAW"
    usage = 'Un-"prepared" data.'
    requirement = PHU({'{prohibit}GEM-TLM': ".*?" })
    
newtypes.append(RAW())
