
GMOS_CAL Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_CAL

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_CAL.py

.. code-block:: python
    :linenos:

    class GMOS_CAL(DataClassification):
        name="GMOS_CAL"
        usage = """
            Applies to all calibration datasets from the GMOS instruments
            """
        parent = "GMOS"
        requirement = ISCLASS("GMOS") & OR([  ISCLASS("GMOS_IMAGE_FLAT"),
                                              ISCLASS("GMOS_IMAGE_TWILIGHT"),
                                              ISCLASS("GMOS_BIAS"),
                                              ISCLASS("GMOS_LS_FLAT"),
                                              ISCLASS("GMOS_LS_TWILIGHT"),
                                              ISCLASS("GMOS_LS_ARC"),
                                              ISCLASS("GMOS_MOS_FLAT"),
                                              ISCLASS("GMOS_MOS_TWILIGHT"),
                                              ISCLASS("GMOS_MOS_ARC"),
                                              ISCLASS("GMOS_IFU_FLAT"),
                                              ISCLASS("GMOS_IFU_TWILIGHT"),
                                              ISCLASS("GMOS_IFU_ARC")  ])
    
    newtypes.append(GMOS_CAL())



