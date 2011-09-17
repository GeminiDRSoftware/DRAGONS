
MICHELLE, MICHELLE_IMAGE, MICHELLE_SPECT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classifications
    MICHELLE, MICHELLE_IMAGE, MICHELLE_SPECT

Source Location 
    ADCONFIG_Gemini/classifications/types/MICHELLE/gemdtype.MICHELLE.py

.. code-block:: python
    :linenos:

    
    class MICHELLE(DataClassification):
        name="MICHELLE"
        usage = "Applies to datasets from the MICHELLE instrument"
        parent = "GEMINI"
        requirement = PHU(INSTRUME= 'michelle')
    
    newtypes.append(MICHELLE())
    
    class MICHELLE_IMAGE(DataClassification):
        name="MICHELLE_IMAGE"
        usage = "Applies to IMAGE datasets from the MICHELLE instrument"
        parent = "MICHELLE"
        requirement = ISCLASS("MICHELLE") & PHU(CAMERA= 'imaging')
    
    newtypes.append(MICHELLE_IMAGE())
    
    class MICHELLE_SPECT(DataClassification):
        name="MICHELLE_SPECT"
        usage = "Applies to IMAGE datasets from the MICHELLE instrument"
        parent = "MICHELLE"
        requirement = ISCLASS("MICHELLE") & PHU(CAMERA= 'spectroscopy')
    
    newtypes.append(MICHELLE_SPECT())
    
    



