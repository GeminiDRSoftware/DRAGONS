
GMOS_NODANDSHUFFLE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_NODANDSHUFFLE

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_NODANDSHUFFLE.py

.. code-block:: python
    :linenos:

    
    class GMOS_NODANDSHUFFLE(DataClassification):
        name="GMOS_NODANDSHUFFLE"
        usage = ""
        typeReqs= []
        phuReqs= {}
        parent = "GMOS"
        requirement = PHU(NODPIX='.*')
    
    newtypes.append(GMOS_NODANDSHUFFLE())



