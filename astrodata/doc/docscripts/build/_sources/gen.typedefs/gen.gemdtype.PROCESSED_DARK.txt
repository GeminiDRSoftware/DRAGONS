
PROCESSED_DARK Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    PROCESSED_DARK

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.PROCESSED_DARK.py

.. code-block:: python
    :linenos:

    class PROCESSED_DARK(DataClassification):
        
        name="PROCESSED_DARK"
        usage = 'Applies to all dark data stored using storeProcessedDark.'
        parent = "UNPREPARED"
        requirement = PHU( {'{re}.*?PROCDARK': ".*?" })
        
    newtypes.append(PROCESSED_DARK())



