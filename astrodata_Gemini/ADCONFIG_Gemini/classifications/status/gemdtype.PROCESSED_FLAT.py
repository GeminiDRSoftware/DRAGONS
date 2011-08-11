class PROCESSED_FLAT(DataClassification):
    
    name="PROCESSED_FLAT"
    usage = 'Applies to all "giflat"ed flat data, or data stored using storeProcessedFlat.'
    parent = "UNPREPARED"
    requirement = OR([PHU( {'{re}.*?GIFLAT': ".*?" }),
                      PHU( {'{re}.*?PROCFLAT': ".*?" })])
    
newtypes.append(PROCESSED_FLAT())
