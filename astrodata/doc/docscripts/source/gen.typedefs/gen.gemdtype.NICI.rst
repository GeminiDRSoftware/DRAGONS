
NICI Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NICI

Source Location 
    ADCONFIG_Gemini/classifications/types/NICI/gemdtype.NICI.py

.. code-block:: python
    :linenos:

    class NICI(DataClassification):
        name="NICI"
        usage = "Applies to all datasets taken with the NICI instrument."
        parent = "GEMINI"
        requirement = PHU(INSTRUME='NICI')
    
    newtypes.append(NICI())



