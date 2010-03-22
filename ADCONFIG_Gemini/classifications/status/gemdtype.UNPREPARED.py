class UNPREPARED(DataClassification):
    editprotect=True
    name="UNPREPARED"
    usage = 'Un-"prepared" data.'
    requirement= PHU({'{prohibit,re}.*?PREPARE': ".*?" })
    
newtypes.append(UNPREPARED())
