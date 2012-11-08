class OBSERVED(DataClassification):
    editprotect=False
    name="OBSERVED"
    usage = 'Sample type, checkes for OBSERVER header'
    requirement = PHU({'OBSERVER': ".*?" })
   
newtypes.append(OBSERVED())