
PROCESSED_ARC Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    PROCESSED_ARC

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.PROCESSED_ARC.py

.. code-block:: python
    :linenos:

    class PROCESSED_ARC(DataClassification):
        
        name="PROCESSED_ARC"
        usage = 'Applies to all data processed by storeProcessedArc.'
        parent = "UNPREPARED"
        requirement = PHU( {'{re}.*?PROCARC': ".*?" })
        
    newtypes.append(PROCESSED_ARC())



