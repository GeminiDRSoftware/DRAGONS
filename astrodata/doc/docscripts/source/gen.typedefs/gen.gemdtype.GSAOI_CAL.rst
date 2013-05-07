
GSAOI_CAL Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GSAOI_CAL

Source Location 
    ADCONFIG_Gemini/classifications/types/GSAOI/gemdtype.GSAOI_CAL.py

.. code-block:: python
    :linenos:

    class GSAOI_CAL(DataClassification):
        name="GSAOI_CAL"
        usage = """
            Applies to all calibration datasets from the GSAOI instrument
            """
        parent = "GSAOI"
        requirement = ISCLASS("GSAOI") & OR([  ISCLASS("GSAOI_IMAGE_FLAT"),
                                            ISCLASS("GSAOI_IMAGE_TWILIGHT"),
                                            ISCLASS("GSAOI_DARK")  ])
    
    newtypes.append(GSAOI_CAL())



