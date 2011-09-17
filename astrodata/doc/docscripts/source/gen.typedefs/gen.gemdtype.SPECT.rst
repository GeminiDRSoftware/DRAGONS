
SPECT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    SPECT

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.SPECT.py

.. code-block:: python
    :linenos:

    class SPECT(DataClassification):
        name="SPECT"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = """
            Applies to all Gemini spectroscopy datasets
            """
        parent = "GENERIC"
        requirement = OR([  ISCLASS("F2_SPECT"),
                            ISCLASS("GMOS_SPECT"),
                            ISCLASS("GNIRS_SPECT"),
                            ISCLASS("MICHELLE_SPECT"),
                            ISCLASS("NIFS_SPECT"),
                            ISCLASS("NIRI_SPECT"),
                            ISCLASS("TRECS_SPECT")])
    
    newtypes.append(SPECT())



