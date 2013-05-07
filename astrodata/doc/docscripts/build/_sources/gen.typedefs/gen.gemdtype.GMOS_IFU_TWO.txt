
GMOS_IFU_TWO Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_IFU_TWO

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_IFU_TWO.py

.. code-block:: python
    :linenos:

    
    class GMOS_IFU_TWO(DataClassification):
        name="GMOS_IFU_TWO"
        usage = ""
        parent = "GMOS_IFU"
        requirement = ISCLASS('GMOS_IFU') & PHU(MASKNAME='(IFU-2)|(IFU-2-NS)')
    
    newtypes.append(GMOS_IFU_TWO())



