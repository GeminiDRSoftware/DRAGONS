
MICHELLE_SPECT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    MICHELLE_SPECT

Source Location 
    ADCONFIG_Gemini/classifications/types/MICHELLE/gemdtype.MICHELLE_SPECT.py

.. code-block:: python
    :linenos:

    class MICHELLE_SPECT(DataClassification):
        name = "MICHELLE_SPECT"
        usage = """
            Applies to all spectroscopic datasets from the MICHELLE instrument
            """
        parent = "MICHELLE"
        requirement = ISCLASS("MICHELLE") & PHU(CAMERA="spectroscopy")
    
    newtypes.append(MICHELLE_SPECT())



