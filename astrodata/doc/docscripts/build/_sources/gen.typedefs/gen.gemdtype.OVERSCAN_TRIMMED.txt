
OVERSCAN_TRIMMED Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    OVERSCAN_TRIMMED

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.OVERSCAN_TRIMMED.py

.. code-block:: python
    :linenos:

    class OVERSCAN_TRIMMED(DataClassification):
        
        name="OVERSCAN_TRIMMED"
        usage = 'Applies to all overscan trimmed data.'
        parent = "PREPARED"
        requirement = PHU( {'{re}.*?TRIMOVER*?': ".*?" })
        
    newtypes.append(OVERSCAN_TRIMMED())



