
class AZEL_TARGET(DataClassification):
    name="AZEL_TARGET"
    usage = "Data taken on a target in the AZEL_TOPO co-ordinate system"
    
    parent = "GEMINI"
    requirement = PHU(FRAME='AZEL_TOPO') 

newtypes.append(AZEL_TARGET())
