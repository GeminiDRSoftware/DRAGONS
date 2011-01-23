AstroData Class Reference
!!!!!!!!!!!!!!!!!!!!!!!!!

.. toctree::
    
The following is information about the AstroData class. For explanations to
arguments shown for the class constructor, see AstroData.__init__(..).  This
documentation is generated from in source docstrings

.. autoclass:: astrodata.data.AstroData

Basic Functions
@@@@@@@@@@@@@@@

AstroData Constructor
#####################

.. toctree::
    
.. automethod:: astrodata.data.AstroData.__init__

append(..)
###########

.. toctree::

.. automethod:: astrodata.data.AstroData.append

close(..)
############

.. toctree::

.. automethod:: astrodata.data.AstroData.close

insert(..)
############

.. toctree::

.. automethod:: astrodata.data.AstroData.insert

info(..)
#########

.. toctree::

.. automethod:: astrodata.data.AstroData.info

open(..)
#########

.. toctree::

.. automethod:: astrodata.data.AstroData.open

write(..)
############

.. toctree::

.. automethod:: astrodata.data.AstroData.write

Type Information
@@@@@@@@@@@@@@@@@

.. toctree::

.. automethod:: astrodata.data.AstroData.isType

.. automethod:: astrodata.data.AstroData.getTypes

.. automethod:: astrodata.data.AstroData.getStatus

.. automethod:: astrodata.data.AstroData.getTypology

Header Manipulations
@@@@@@@@@@@@@@@@@@@@@@

.. toctree::

Manipulations of headers that relate to high-level metadata covered by
"Astrodata Descriptors" should be read using the appropriate descriptor. 
Descriptions of descriptors are aspects of the Astrodata configuration, so
see the ''Gemini AstroData Type Reference''.

However, to work with meta-data not covered by descriptors, one must read
and write key-value pairs to the HDU headers at the lower-level. AstroData
offers three pairs of functions for getting and setting header values, for
each of three distinct cases.  While it is possible to use the
pyfits.Header directly (available via "ad[..].header"), it is preferrable
to use the AstroData calls which allow AstroData to keep type information
up to date, as well as update any other characteristics maintained in
reference to the dataset's header data.

The three distinct uses are:

+ set/get headers in PHU.
+ set/get headers in the single extension of a single-HDU AstroData
  instance.
+ set/get headers in an extension of a multi-HDU (aka "multi-extension") 
  AstroData instance. This requires specifying the extension index, and
  cannot modify the PHU.

Set/Get PHU Headers
####################

.. toctree::

.. automethod:: astrodata.data.AstroData.phuGetKeyValue

.. automethod:: astrodata.data.AstroData.phuSetKeyValue

Set/Get Single-HDU Headers
###########################

.. toctree::

.. automethod:: astrodata.data.AstroData.getKeyValue

.. automethod:: astrodata.data.AstroData.setKeyValue

Set/Get Multiple-HDU Headers
#############################

.. toctree::

.. automethod:: astrodata.data.AstroData.extGetKeyValue

.. automethod:: astrodata.data.AstroData.extSetKeyValue


Iteration and Subdata
@@@@@@@@@@@@@@@@@@@@@@@

.. toctree::
   
Overview
############
 
.. toctree::
    
    gen.ADMANUAL-ADSubdata.rst
    
countExts(..)
##############

.. toctree::

.. automethod:: astrodata.data.AstroData.countExts


The [] Operator
################

.. toctree::

.. automethod:: astrodata.data.AstroData.__getitem__

Single HDU AstroData Attributes
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

.. toctree::
    :numbered:
    
.. include: gen.ADMANUAL_SingleHDUAD

data attribute
##############
 
.. toctree::
     
.. autoattribute:: astrodata.data.AstroData.data

.. automethod:: astrodata.data.AstroData.getData

.. automethod:: astrodata.data.AstroData.setData

header attribute
###################

.. toctree::

.. autoattribute:: astrodata.data.AstroData.header

.. automethod:: astrodata.data.AstroData.getHeader

.. automethod:: astrodata.data.AstroData.setHeader

Renaming an Extension
######################

.. toctree::

.. automethod:: astrodata.data.AstroData.renameExt

Accessing Pyfits and Numpy Objects
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

.. toctree::
    
    gen.ADMANUAL-AccessingPyfitsOjbects

Module Level Functions
@@@@@@@@@@@@@@@@@@@@@@@@

.. toctree::
    
correlate(..)
#############

.. toctree::
    
.. autofunction:: astrodata.data.correlate

prepOutput(..)
################

.. toctree::

.. autofunction:: astrodata.data.prepOutput

reHeaderKeys(..)
#################

.. toctree::

.. autofunction:: astrodata.data.reHeaderKeys
