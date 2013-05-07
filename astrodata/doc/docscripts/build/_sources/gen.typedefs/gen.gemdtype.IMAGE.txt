
IMAGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    IMAGE

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.IMAGE.py

.. code-block:: python
    :linenos:

    class IMAGE(DataClassification):
        name="IMAGE"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = """
            Applies to all Gemini imaging datasets
            """
        parent = "GENERIC"
        requirement = OR([  ISCLASS("F2_IMAGE"),
                            ISCLASS("GMOS_IMAGE"),
                            ISCLASS("GNIRS_IMAGE"),
                            ISCLASS("GSAOI_IMAGE"),
                            ISCLASS("MICHELLE_IMAGE"),
                            ISCLASS("NICI_IMAGE"),
                            ISCLASS("NIFS_IMAGE"),
                            ISCLASS("NIRI_IMAGE"),
                            ISCLASS("TRECS_IMAGE")])
    
    newtypes.append(IMAGE())



