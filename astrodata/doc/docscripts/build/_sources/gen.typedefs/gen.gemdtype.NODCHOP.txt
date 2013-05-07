
NODCHOP Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NODCHOP

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.NODCHOP.py

.. code-block:: python
    :linenos:

    class NODCHOP(DataClassification):
        name="NODCHOP"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = '''
            Applies to data marked with NOD and CHOP keywords.
            TEST TYPE!!!
            Made to test Structures Feature.
            '''
        requirement = PHU(DATATYPE="marked-nodandchop")
    
    newtypes.append( NODCHOP())



