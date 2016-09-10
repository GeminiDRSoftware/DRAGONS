class PROCESSED_DARK(DataClassification):
    
    name="PROCESSED_DARK"
    usage = 'Applies to all dark data stored using storeProcessedDark.'
    parent = "UNPREPARED"
    requirement = PHU( {'{re}.*?PROCDARK': ".*?" })
    
newtypes.append(PROCESSED_DARK())
