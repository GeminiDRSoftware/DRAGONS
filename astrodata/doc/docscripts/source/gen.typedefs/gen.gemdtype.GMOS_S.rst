
GMOS_S Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_S

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_S.py

.. code-block:: python
    :linenos:

    
    class GMOS_S(DataClassification):
        name="GMOS_S"
        usage = "For data from GMOS South"
        
        parent = "GMOS"
        requirement = PHU(INSTRUME='GMOS-S')
    
    newtypes.append(GMOS_S())



