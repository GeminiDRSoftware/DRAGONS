
GMOS_MOS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_MOS

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_MOS.py

.. code-block:: python
    :linenos:

    class GMOS_MOS(DataClassification):
        name="GMOS_MOS"
        usage = """
            Applies to all MOS datasets from the GMOS instruments
            """
        parent = "GMOS_SPECT"
        requirement = AND([  ISCLASS("GMOS_SPECT"),
                             PHU(MASKTYP="1"),
                             PHU({"{prohibit}MASKNAME": ".*arcsec"}),
                             PHU({"{prohibit}MASKNAME": "IFU*"})  ])
    
    newtypes.append(GMOS_MOS())



