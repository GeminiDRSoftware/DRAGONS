
PROCESSED_FRINGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    PROCESSED_FRINGE

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.PROCESSED_FRINGE.py

.. code-block:: python
    :linenos:

    class PROCESSED_FRINGE(DataClassification):
        
        name="PROCESSED_FRINGE"
        usage = 'Applies to all "gifringe"ed data.'
        parent = "UNPREPARED"
        requirement = OR([PHU( {'{re}.*?GIFRINGE': ".*?" }),
                          PHU( {'{re}.*?PROCFRNG': ".*?" })])
        
    newtypes.append(PROCESSED_FRINGE())



