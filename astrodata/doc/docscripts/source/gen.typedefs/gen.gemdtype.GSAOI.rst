
GSAOI Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GSAOI

Source Location 
    ADCONFIG_Gemini/classifications/types/GSAOI/gemdtype.GSAOI.py

.. code-block:: python
    :linenos:

    
    class GSAOI(DataClassification):
        name="GSAOI"
        usage = "Applies to any data from the GSAOI instrument."
        parent = "GEMINI"
    
        requirement = PHU(INSTRUME='GSAOI')
    
    newtypes.append(GSAOI())



