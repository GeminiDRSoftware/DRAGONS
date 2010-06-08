class PREPARED(DataClassification):
    
    name="PREPARED"
    usage = 'Applies to all "prepared" data.'
    parent = "UNPREPARED"
    requirement = PHU( {'{re}.*?PREPARE': ".*?" })
    
newtypes.append(PREPARED())
