class PREPARED(DataClassification):
    editprotect=True
    name="PREPARED"
    usage = '"prepared" data.'
    requiement = PHU( {'{re}.*?PREPARE': ".*?" })
    
newtypes.append(PREPARED())
