class UNPREPARED(DataClassification):
    editprotect=True
    name="UNPREPARED"
    usage = 'Applies to un-"prepared" datasets, datasets which have not had the prepare task run on them.'
    parent = "RAW"
    requirement= PHU({'{prohibit,re}.*?PREPAR*?': ".*?" })
    
newtypes.append(UNPREPARED())
