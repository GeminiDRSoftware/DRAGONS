
GSAOI_IMAGE_FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GSAOI_IMAGE_FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/GSAOI/gemdtype.GSAOI_IMAGE_FLAT.py

.. code-block:: python
    :linenos:

    class GSAOI_IMAGE_FLAT(DataClassification):
        name="GSAOI_IMAGE_FLAT"
        usage = """
            Applies to all imaging flat datasets from the GSAOI instrument
            """
        parent = "GSAOI_IMAGE"
        requirement = AND([  ISCLASS("GSAOI_IMAGE"),
                             OR([  PHU(OBSTYPE="FLAT"),
                                   OR([  PHU(OBJECT="Twilight"),
                                         PHU(OBJECT="twilight")  ])  ])  ])
    
    newtypes.append(GSAOI_IMAGE_FLAT())



