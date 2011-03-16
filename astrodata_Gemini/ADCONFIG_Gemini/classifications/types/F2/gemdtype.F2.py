class F2(DataClassification):
    name="F2"
    usage = """
        Applies to all datasets from the FLAMINGOS-2 instrument
        """
    parent = "GEMINI"
    # Commissioning data from 28 August 2009 to 20 February 2010 use "Flam" as
    # the value for the INSTRUME keyword. The final value for the INSTRUME
    # keyword will be "F2".
    requirement = OR([  PHU(INSTRUME="Flam"),
                        PHU(INSTRUME="F2")  ])

newtypes.append(F2())
