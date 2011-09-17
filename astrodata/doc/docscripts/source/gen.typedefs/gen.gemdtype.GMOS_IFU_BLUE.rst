
GMOS_IFU_BLUE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_IFU_BLUE

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_IFU_BLUE.py

.. code-block:: python
    :linenos:

    
    class GMOS_IFU_BLUE(DataClassification):
        name="GMOS_IFU_BLUE"
        usage = ""
        parent = "GMOS_IFU"
        requirement = ISCLASS('GMOS_IFU') & PHU(MASKNAME='(IFU-B)|(IFU-B-NS)')
    
    newtypes.append(GMOS_IFU_BLUE())



