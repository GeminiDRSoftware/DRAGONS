
BIAS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    BIAS

Source Location 
    ADCONFIG_Gemini/classifications/types/generic/gemdtype.BIAS.py

.. code-block:: python
    :linenos:

    class BIAS(DataClassification):
        name="BIAS"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = '''
            Applies to any Gemini dataset which is an instrument bias calibration.'''
        parent = "CAL"
        requirement = ISCLASS("GMOS_BIAS")
    
    newtypes.append( BIAS())



