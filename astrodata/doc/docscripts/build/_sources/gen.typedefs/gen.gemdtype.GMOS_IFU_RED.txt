
GMOS_IFU_RED Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_IFU_RED

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_IFU_RED.py

.. code-block:: python
    :linenos:

    
    class GMOS_IFU_RED(DataClassification):
        name="GMOS_IFU_RED"
        usage = ""
        parent = "GMOS_IFU"
        requirement = ISCLASS('GMOS_IFU') & PHU(MASKNAME='(IFU-R)|(IFU-R-NS)')
    
    newtypes.append(GMOS_IFU_RED())



