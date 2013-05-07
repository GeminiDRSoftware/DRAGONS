
F2_IMAGE_FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    F2_IMAGE_FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/F2/gemdtype.F2_IMAGE_FLAT.py

.. code-block:: python
    :linenos:

    class F2_IMAGE_FLAT(DataClassification):
        name="F2_IMAGE_FLAT"
        usage = """
            Applies to all imaging flat datasets from the FLAMINGOS-2 instrument
            """
        parent = "F2_IMAGE"
        requirement = AND([  ISCLASS("F2_IMAGE"),
                             OR([  PHU(OBSTYPE="FLAT"),
                                   OR([  PHU(OBJECT="Twilight"),
                                         PHU(OBJECT="twilight")  ])  ])  ])
    
    newtypes.append(F2_IMAGE_FLAT())



