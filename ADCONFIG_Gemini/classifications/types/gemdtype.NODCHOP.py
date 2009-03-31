class NODCHOP(DataClassification):
    name="NODCHOP"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to data marked with NOD and CHOP keywords
        '''
    phuReqs = { "DATATYPE" : "marked-nodandchop"}

newtypes.append( NODCHOP())
