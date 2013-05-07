
NIRI_IMAGE_TWILIGHT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NIRI_IMAGE_TWILIGHT

Source Location 
    ADCONFIG_Gemini/classifications/types/NIRI/gemdtype.NIRI_IMAGE_TWILIGHT.py

.. code-block:: python
    :linenos:

    class NIRI_IMAGE_TWILIGHT(DataClassification):
        name="NIRI_IMAGE_TWILIGHT"
        usage = """
            Applies to all imaging twilight flat datasets from the NIRI
            instrument
            """
        parent = "NIRI_IMAGE_FLAT"
        requirement = AND([  ISCLASS("NIRI_IMAGE_FLAT"),
                             OR([  PHU(OBJECT="Twilight"),
                                   PHU(OBJECT="twilight")  ])  ])
    
    newtypes.append(NIRI_IMAGE_TWILIGHT())



