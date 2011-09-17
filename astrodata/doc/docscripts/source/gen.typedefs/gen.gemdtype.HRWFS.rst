
HRWFS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    HRWFS

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.HRWFS.py

.. code-block:: python
    :linenos:

    
    class HRWFS(DataClassification):
        name="HRWFS"
        usage = "Applies to all datasets from the HRWFS instrument."
        parent = "GEMINI"
        requirement = PHU(INSTRUME='hrwfs')
    
    newtypes.append(HRWFS())



