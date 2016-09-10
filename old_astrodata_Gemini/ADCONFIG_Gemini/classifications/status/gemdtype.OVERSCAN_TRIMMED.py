class OVERSCAN_TRIMMED(DataClassification):
    
    name="OVERSCAN_TRIMMED"
    usage = 'Applies to all overscan trimmed data.'
    parent = "PREPARED"
    requirement = PHU( {'{re}.*?TRIMOVER*?': ".*?" })
    
newtypes.append(OVERSCAN_TRIMMED())
