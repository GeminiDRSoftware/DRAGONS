
F2_SPECT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    F2_SPECT

Source Location 
    ADCONFIG_Gemini/classifications/types/F2/gemdtype.F2_SPECT.py

.. code-block:: python
    :linenos:

    class F2_SPECT(DataClassification):
        name="F2_SPECT"
        usage = """
            Applies to all spectroscopic datasets from the FLAMINGOS-2 instrument
            """
        parent = "F2"
        requirement = AND([  ISCLASS("F2"),
                             OR([  PHU(DCKERPOS="Long_slit"),
                                   PHU(MOSPOS=".?pix-slit"),
                                   PHU(DCKERPOS="mos"),
                                   PHU(MOSPOS="mos.?"),  ]),
                             NOT(ISCLASS("F2_DARK"))  ])
    
    newtypes.append(F2_SPECT())



