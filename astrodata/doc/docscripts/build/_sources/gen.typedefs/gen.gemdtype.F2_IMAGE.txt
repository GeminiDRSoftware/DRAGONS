
F2_IMAGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    F2_IMAGE

Source Location 
    ADCONFIG_Gemini/classifications/types/F2/gemdtype.F2_IMAGE.py

.. code-block:: python
    :linenos:

    class F2_IMAGE(DataClassification):
        name="F2_IMAGE"
        usage = """
            Applies to all imaging datasets from the FLAMINGOS-2 instrument
            """
        parent = "F2"
        # Commissioning data from 28 August 2009 to 20 February 2010 use the
        # MASKNAME keyword to specify whether the data is imaging, longslit or
        # mos. The final keyword to use will be DCKERPOS or MOSPOS.
        requirement = AND([  ISCLASS("F2"),
                             OR([  PHU(MASKNAME="imaging"),
                                   PHU(DECKER="Open"),
                                   PHU(MOSPOS="Open")  ]),
                             NOT(ISCLASS("F2_DARK"))  ])
    
    newtypes.append(F2_IMAGE())



