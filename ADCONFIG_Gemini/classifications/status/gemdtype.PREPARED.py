class PREPARED(DataClassification):
    editprotect=True
    name="PREPARED"
    usage = 'Applies to all "prepared" data.'
    parent = "UNPREPARED"
    requiement = PHU( {'{re}.*?PREPARE': ".*?" })
    
newtypes.append(PREPARED())
