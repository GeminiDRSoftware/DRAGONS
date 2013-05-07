
NIFS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NIFS

Source Location 
    ADCONFIG_Gemini/classifications/types/NIFS/gemdtype.NIFS.py

.. code-block:: python
    :linenos:

    
    class NIFS(DataClassification):
        name="NIFS"
        usage = "Applies to datasets from NIFS instrument"
        parent = "GEMINI"
        requirement = PHU(INSTRUME='NIFS')
    
    newtypes.append(NIFS())



