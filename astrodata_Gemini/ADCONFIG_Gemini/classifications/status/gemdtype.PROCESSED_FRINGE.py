class PROCESSED_FRINGE(DataClassification):
    
    name="PROCESSED_FRINGE"
    usage = 'Applies to all "gifringe"ed data.'
    parent = "UNPREPARED"
    requirement = PHU( {'{re}.*?GIFRINGE': ".*?" })
    
newtypes.append(PROCESSED_FRINGE())
