
PROCESSED_BIAS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    PROCESSED_BIAS

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.PROCESSED_BIAS.py

.. code-block:: python
    :linenos:

    class PROCESSED_BIAS(DataClassification):
        
        name="PROCESSED_BIAS"
        usage = 'Applies to all "gbias"ed data.'
        parent = "UNPREPARED"
        requirement = OR([PHU( {'{re}.*?GBIAS': ".*?" }),
                          PHU( {'{re}.*?PROCBIAS': ".*?" })])
        
    newtypes.append(PROCESSED_BIAS())



