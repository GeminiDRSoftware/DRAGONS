
RAW Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    RAW

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.RAW.py

.. code-block:: python
    :linenos:

    class RAW(DataClassification):
        editprotect=True
        name="RAW"
        usage = 'Applies to data that has not been modified by the gemini package in any way (looks for GEM-TLM stamp in header).'
        requirement = PHU({'{prohibit}GEM-TLM': ".*?" })
        
    newtypes.append(RAW())



