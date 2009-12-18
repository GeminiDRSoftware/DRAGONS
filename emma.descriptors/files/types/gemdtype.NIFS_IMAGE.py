
class NIFS_IMAGE(DataClassification):
    name="NIFS_IMAGE"
    usage = ""
    typeReqs= ['NIFS']
    phuReqs= {'FILTER3': '(.*?)pupil(.*?)'}

newtypes.append(NIFS_IMAGE())
