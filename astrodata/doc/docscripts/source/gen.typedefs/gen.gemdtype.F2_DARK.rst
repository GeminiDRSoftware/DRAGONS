
F2_DARK Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    F2_DARK

Source Location 
    ADCONFIG_Gemini/classifications/types/F2/gemdtype.F2_DARK.py

.. code-block:: python
    :linenos:

    class F2_DARK(DataClassification):
        name="F2_DARK"
        usage = """
            Applies to all dark datasets from the FLAMINGOS-2 instrument
            """
        parent = "F2_IMAGE"
        requirement = AND([  ISCLASS("F2_IMAGE"),
                             PHU(OBSTYPE="DARK")  ])
    
    newtypes.append(F2_DARK())



