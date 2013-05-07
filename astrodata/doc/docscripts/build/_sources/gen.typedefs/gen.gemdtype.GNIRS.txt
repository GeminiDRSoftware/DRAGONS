
GNIRS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GNIRS

Source Location 
    ADCONFIG_Gemini/classifications/types/GNIRS/gemdtype.GNIRS.py

.. code-block:: python
    :linenos:

    
    class GNIRS(DataClassification):
        name="GNIRS"
        usage = "Applies to all datasets from the GNIRS instrument."
        parent = "GEMINI"
        requirement = PHU(INSTRUME='GNIRS')
    
    newtypes.append(GNIRS())



