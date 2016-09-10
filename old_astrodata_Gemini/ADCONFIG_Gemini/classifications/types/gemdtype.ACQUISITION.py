class ACQUISITION(DataClassification):
    name="ACQUISITION"
    usage = """
        Applies to all Gemini acquisitions
        """
    parent = "GENERIC"
    requirement = OR([  PHU(OBSCLASS="acq"),
                        PHU(OBSCLASS="acqCal")  ])

newtypes.append(ACQUISITION())
