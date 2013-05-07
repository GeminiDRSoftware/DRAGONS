
CAL Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    CAL

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.CAL.py

.. code-block:: python
    :linenos:

    class CAL(DataClassification):
        name="CAL"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = """
            Special parent to group generic types (e.g. IMAGE, SPECT, MOS, IFU)
            """
        parent = "GENERIC"
        requirement = OR([  ISCLASS("F2_CAL"),
                            ISCLASS("GMOS_CAL"),
                            ISCLASS("GSAOI_CAL"),
                            ISCLASS("NICI_CAL")])
    
    newtypes.append(CAL())



