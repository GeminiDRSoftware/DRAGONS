class MARKED(DataClassification):
    editprotect=False
    name="MARKED"
    usage = 'Sample type, checks "S_MARKED" header'
    requirement = PHU({'S_MARKED': ".*?" })
   
newtypes.append(MARKED())