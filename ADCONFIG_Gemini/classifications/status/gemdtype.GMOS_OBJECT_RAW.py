class GMOS_OBJECT_RAW(DataClassification):
    name="GMOS_OBJECT_RAW"
    usage = 'Un-"prepared" GMOS "object" data.'
    requirements = ISCLASS('GMOS_RAW','GMOS_OBJECT')
    
newtypes.append(GMOS_OBJECT_RAW())
