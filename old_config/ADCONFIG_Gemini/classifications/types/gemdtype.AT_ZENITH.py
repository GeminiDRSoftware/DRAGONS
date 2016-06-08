class AT_ZENITH(DataClassification):
    name="AT_ZENITH"
    usage = "Data taken at Zenith in the AZEL_TOPO co-ordinate system. Normally this means that the telescope was parked at zenith."
    
    parent = "AZEL_TARGET"
    requirement = PHU(FRAME='AZEL_TOPO') & PHU(ELEVATIO='90.*')

newtypes.append(AT_ZENITH())
