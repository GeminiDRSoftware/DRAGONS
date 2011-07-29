class PROCESSED_FLAT(DataClassification):
    
    name="PROCESSED_FLAT"
    usage = 'Applies to all "giflat"ed or "normalize_flat_image_gmos"ed data.'
    parent = "UNPREPARED"
    requirement = OR([PHU( {'{re}.*?GIFLAT': ".*?" }),
                      PHU( {'{re}.*?NORMFLAT': ".*?" })])
    
newtypes.append(PROCESSED_FLAT())
