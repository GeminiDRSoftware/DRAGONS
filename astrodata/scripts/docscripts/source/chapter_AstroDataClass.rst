AstroData Class Reference
!!!!!!!!!!!!!!!!!!!!!!!!!

The following is class information from the docstrings.

.. autoclass:: astrodata.data.AstroData

Basic Functions
@@@@@@@@@@@@@@@

.. automethod:: astrodata.data.AstroData.__init__

.. automethod:: astrodata.data.AstroData.append

.. automethod:: astrodata.data.AstroData.open

Iteration and Subdata
############################

.. toctree::
    :numbered:
    
.. include:: gen.ADMANUAL-ADSubdata.rst

The [] Operator
$$$$$$$$$$$$$$$

.. automethod:: astrodata.data.AstroData.__getitem__

Single HDU AstroData Methods
############################

.. toctree:: 
    :numbered:
    
    gen.ADMANUAL_SingleHDUAD
    
.. autoattribute:: astrodata.data.AstroData.data

.. autoattribute:: astrodata.data.AstroData.header

Module Level Functions
@@@@@@@@@@@@@@@@@@@@@@@@
    
.. autofunction:: astrodata.data.correlate

.. autofunction:: astrodata.data.prepOutput

.. autofunction:: astrodata.data.reHeaderKeys
