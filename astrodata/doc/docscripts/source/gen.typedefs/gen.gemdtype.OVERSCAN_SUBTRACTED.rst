
OVERSCAN_SUBTRACTED Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    OVERSCAN_SUBTRACTED

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.OVERSCAN_SUBTRACTED.py

.. code-block:: python
    :linenos:

    class OVERSCAN_SUBTRACTED(DataClassification):
        
        name="OVERSCAN_SUBTRACTED"
        usage = 'Applies to all overscan subtracted data.'
        parent = "PREPARED"
        requirement = PHU( {'{re}.*?SUBOVER*?': ".*?" })
        
    newtypes.append(OVERSCAN_SUBTRACTED())



