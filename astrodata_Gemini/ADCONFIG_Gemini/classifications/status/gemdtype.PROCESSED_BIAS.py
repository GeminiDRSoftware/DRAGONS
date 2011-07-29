class PROCESSED_BIAS(DataClassification):
    
    name="PROCESSED_BIAS"
    usage = 'Applies to all "gbias"ed data.'
    parent = "UNPREPARED"
    requirement = OR([PHU( {'{re}.*?GBIAS': ".*?" }),
                      AND([PHU( {'{re}.*?STACK': ".*?" }),
                           ISCLASS("GMOS_BIAS")])])
    
newtypes.append(PROCESSED_BIAS())
