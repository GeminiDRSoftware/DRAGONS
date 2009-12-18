class NIFS_RAW(DataClassification):
    editprotect=True
    name="NIFS_RAW"
    usage = 'Un-"prepared" NIFS data.'
    typeReqs= ['NIFS']
    phuReqs= {'{prohibit}NFPREPARE': ".*?" }
    
newtypes.append(NIFS_RAW())
