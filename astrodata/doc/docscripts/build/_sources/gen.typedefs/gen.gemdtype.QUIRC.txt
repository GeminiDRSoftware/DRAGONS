
HOKUPAAQUIRC Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    HOKUPAAQUIRC

Source Location 
    ADCONFIG_Gemini/classifications/types/QUIRC/gemdtype.QUIRC.py

.. code-block:: python
    :linenos:

    
    class HOKUPAAQUIRC(DataClassification):
        name="HOKUPAAQUIRC"
        usage = "Applies to datasets from the HOKUPAA+QUIRC instrument"
        parent = "GEMINI"
        requirement = PHU(INSTRUME='Hokupaa\+QUIRC')
    
    newtypes.append(HOKUPAAQUIRC())



