class FLAMINGOS2_RAW(DataClassification):
    editprotect=True
    name="FLAMINGOS2_RAW"
    usage = 'Un-"prepared" FLAMINGOS2 data.'
    typeReqs= ['FLAMINGOS2']
    phuReqs= {'{prohibit}F2PREPARE': ".*?" }
    
newtypes.append(FLAMINGOS2_RAW())
