class OVERSCAN_SUBTRACTED(DataClassification):
    
    name="OVERSCAN_SUBTRACTED"
    usage = 'Applies to all overscan subtracted data.'
    parent = "PREPARED"
    requirement = PHU( {'{re}.*?SUBOVER*?': ".*?" })
    
newtypes.append(OVERSCAN_SUBTRACTED())
