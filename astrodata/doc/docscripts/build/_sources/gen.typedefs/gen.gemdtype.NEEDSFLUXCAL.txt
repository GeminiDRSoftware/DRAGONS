
NEEDSFLUXCAL Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NEEDSFLUXCAL

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.NEEDSFLUXCAL.py

.. code-block:: python
    :linenos:

    class NEEDSFLUXCAL(DataClassification):
        editprotect=False
        name="NEEDSFLUXCAL"
        usage = 'Applies to data ready for flux calibration.'
        requirement = ISCLASS("IMAGE") & ISCLASS("PREPARED")
        
    newtypes.append(NEEDSFLUXCAL())



