class UNMARKED(DataClassification):
    editprotect=False
    name="UNMARKED"
    usage = 'No comment.'
    requirement = PHU({'{prohibit}S_MARKED': ".*?" })
   
newtypes.append(UNMARKED())