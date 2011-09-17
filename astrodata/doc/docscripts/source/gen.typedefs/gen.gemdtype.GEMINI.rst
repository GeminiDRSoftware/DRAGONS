
GEMINI Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GEMINI

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.GEMINI.py

.. code-block:: python
    :linenos:

    class GEMINI(DataClassification):
        name="GEMINI"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = '''
            Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
            '''
        requirement = OR(ISCLASS("GEMINI_NORTH"),
                         ISCLASS("GEMINI_SOUTH"))
    
    newtypes.append( GEMINI())



