
GMOS_IMAGE_FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_IMAGE_FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_IMAGE_FLAT.py

.. code-block:: python
    :linenos:

    class GMOS_IMAGE_FLAT(DataClassification):
        name="GMOS_IMAGE_FLAT"
        usage = """
            Applies to all imaging flat datasets from the GMOS instruments
            """
        parent = "GMOS_IMAGE"
        requirement = AND(NOT(PHU({"{re}FILTER.*?": "Hartmann.*?"})),
                          OR(AND([  ISCLASS("GMOS_IMAGE"),
                                    PHU(OBSTYPE="FLAT")       ]),
                             AND([  ISCLASS("GMOS_IMAGE"),
                                    OR([PHU(OBJECT="Twilight"),
                                        PHU(OBJECT="twilight")])  ])))
    
    newtypes.append(GMOS_IMAGE_FLAT())



