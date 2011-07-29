class PROCESSED_FRINGE(DataClassification):
    
    name="PROCESSED_FRINGE"
    usage = 'Applies to all "gifringe"ed data.'
    parent = "UNPREPARED"
    requirement = OR([PHU( {'{re}.*?GIFRINGE': ".*?" }),
                      PHU( {'{re}.*?PROCFRNG': ".*?" })])
    
newtypes.append(PROCESSED_FRINGE())
