class PROCESSED_FLAT(DataClassification):
    
    name="PROCESSED_FLAT"
    usage = 'Applies to all "giflat"ed or "normalize_image"d flat data.'
    parent = "UNPREPARED"
    requirement = OR([PHU( {'{re}.*?GIFLAT': ".*?" }),
                      PHU( {'{re}.*?PROCFLAT': ".*?" })])
    
newtypes.append(PROCESSED_FLAT())
