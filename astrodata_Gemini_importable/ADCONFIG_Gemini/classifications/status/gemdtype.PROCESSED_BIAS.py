class PROCESSED_BIAS(DataClassification):
    
    name="PROCESSED_BIAS"
    usage = 'Applies to all "gbias"ed data.'
    parent = "UNPREPARED"
    requirement = OR([PHU( {'{re}.*?GBIAS': ".*?" }),
                      PHU( {'{re}.*?PROCBIAS': ".*?" })])
    
newtypes.append(PROCESSED_BIAS())
