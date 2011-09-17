
F2_LS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    F2_LS

Source Location 
    ADCONFIG_Gemini/classifications/types/F2/gemdtype.F2_LS.py

.. code-block:: python
    :linenos:

    class F2_LS(DataClassification):
        name="F2_LS"
        usage = """
            Applies to all longslit datasets from the FLAMINGOS-2 instrument
            """
        parent = "F2_SPECT"
        requirement = ISCLASS("F2_SPECT") & OR([  PHU(DCKERPOS="Long_slit"),
                                                  PHU(MOSPOS=".?pix-slit")  ])
    
    newtypes.append(F2_LS())
     



