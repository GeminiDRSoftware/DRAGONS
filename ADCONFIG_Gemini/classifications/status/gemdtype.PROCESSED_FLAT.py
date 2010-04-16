class PROCESSED_FLAT(DataClassification):
    
    name="PROCESSED_FLAT"
    usage = 'Applies to all "giflat"ed data.'
    parent = "UNPREPARED"
    requirement = PHU( {'{re}.*?GIFLAT': ".*?" })
    
newtypes.append(PROCESSED_FLAT())
