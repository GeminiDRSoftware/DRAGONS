class PROCESSED_SCIENCE(DataClassification):
    
    name="PROCESSED_SCIENCE"
    usage = 'Attempts to identify processed science data.'
    parent = "UNPREPARED"
    requirement = AND([PHU( {'{re}.*?IMCOADD': ".*?" }),
                       PHU( {'{re}.*?PREPAR*?': ".*?" }),
                       PHU(OBSTYPE='OBJECT')])
    
newtypes.append(PROCESSED_SCIENCE())
