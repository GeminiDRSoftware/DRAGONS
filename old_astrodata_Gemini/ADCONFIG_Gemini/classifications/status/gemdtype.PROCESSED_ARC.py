class PROCESSED_ARC(DataClassification):
    
    name="PROCESSED_ARC"
    usage = 'Applies to all data processed by storeProcessedArc.'
    parent = "UNPREPARED"
    requirement = PHU( {'{re}.*?PROCARC': ".*?" })
    
newtypes.append(PROCESSED_ARC())
