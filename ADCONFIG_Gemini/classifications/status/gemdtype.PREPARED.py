class PREPARED(DataClassification):
    editprotect=True
    name="PREPARED"
    usage = '"prepared" data.'
    phuReqs= {'{re}.*?PREPARE': ".*?" }
    
newtypes.append(PREPARED())
