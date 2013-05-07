
NIRI_IMAGE_FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NIRI_IMAGE_FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/NIRI/gemdtype.NIRI_IMAGE_FLAT.py

.. code-block:: python
    :linenos:

    class NIRI_IMAGE_FLAT(DataClassification):
        name="NIRI_IMAGE_FLAT"
        usage = """
            Applies to all imaging flat datasets from the NIRI instrument
            """
        parent = "NIRI_IMAGE"
        requirement = AND([  ISCLASS("NIRI_IMAGE"),
                             OR([  PHU(OBSTYPE="FLAT"),
                                   OR([  PHU(OBJECT="Twilight"),
                                         PHU(OBJECT="twilight")  ])  ])  ])
    
    newtypes.append(NIRI_IMAGE_FLAT())



