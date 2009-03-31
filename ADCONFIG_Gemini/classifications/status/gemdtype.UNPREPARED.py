class UNPREPARED(DataClassification):
    editprotect=True
    name="UNPREPARED"
    usage = 'Un-"prepared" data.'
    phuReqs= {'{prohibit,re}.*?PREPARE': ".*?" }
    
newtypes.append(UNPREPARED())
