class PROCESSED_BIAS(DataClassification):
    
    name="PROCESSED_BIAS"
    usage = 'Applies to all "gbias"ed data.'
    parent = "UNPREPARED"
    requirement = PHU( {'{re}.*?GBIAS': ".*?" })
    
newtypes.append(PROCESSED_BIAS())
