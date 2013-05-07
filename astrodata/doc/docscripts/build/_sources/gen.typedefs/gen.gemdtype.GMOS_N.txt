
GMOS_N Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_N

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_N.py

.. code-block:: python
    :linenos:

    class GMOS_N(DataClassification):
        name="GMOS_N"
        usage = ""
        typeReqs= []
        phuReqs= {}
        parent = "GMOS"
        requirement = PHU(INSTRUME='GMOS-N')
    
    newtypes.append(GMOS_N())



