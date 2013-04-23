.. io:

***************************
Input and Output Operations
***************************

Open Existing MEF Files
=======================

An AstroData object can be created from the name of the file on disk or from 
PyFITS HDUList.  An existing MEF file can be open as an AstroData object 
in ``readonly``, ``update``, or ``append`` mode.  The default is ``readonly``.

*(KL: why would anyone want to create an AD from another AD??!!)*
*(KL: what's the deal with store and storeClobber?  Incomprehensible.)*

Here is a very simple example on how to open a file in ``readonly`` mode, 
check the structure, and then close it::

  from astrodata import AstroData
  
  ad = AstroData('N20111124S0203.fits')
  ad.info()
  ad.close()

To open the file in a mode other than ``readonly``, specify the value of the
``mode`` argument::

  ad = AstroData('N20111124S0203.fits', mode='update')

  


Update Existing MEF Files
=========================
To update an existing MEF file, it must have been opened in the ``update`` mode.  Then a collection
of methods can be applied to the AstroData object.  Here we give examples on how to append an 
extension, how to insert an extension, how to remove an extension, and how to replace an extension.
Then we show how do basic arithmetics on the pixel data and the headers in a loop.  Manipulations
of the pixel data and of the headers are covered in more details in later sections (?? and ??, respectively).
Finally we show how to write the updated AstroData object to disk as MEF file. ::

   from astrodata import AstroData
   
   # Open the file to update
   ad = AstroData('N20110313S0188.fits', mode='update')
   ad.info()
   
   # Get an already formed extension from another file (just for the
   # sake of keeping the example simple)
   adread = AstroData('N20110316S0321.fits', mode='readonly')
   new_extension = adread["SCI",2]
   
   # Append an extension.
   # WARNING: new_extension has EXTNAME=SCI and EXTVER=2
   #          ad already has an extension SCI,2.
   #          To avoid conflict, the appended extension needs
   #          to be renumbered to SCI,4. auto_number=True takes
   #          care of that.
   # WARNING: renumbering the appended extension will affect 
   #          adread as new_extension is just a pointer to that
   #          extension in adread.  To avoid the modification of
   #          adread, one either does the deepcopy before the
   #          call to append, or set the do_deepcopy argument
   #          to True, as we do here.
   ad.append(new_extension,auto_number=True,do_deepcopy=True)
   ad.info()
   
   # Insert an extension between two already existing extensions.
   #
   # Let's first rename the new_extension to make it stand out once
   # inserted.
   new_extension = adread['SCI',1]
   new_extension.rename_ext('VAR')
   new_extension.info()
   
   #   Here we insert the extension between the PHU and the first
   #   extension.
   #   WARNING: An AstroData object is a PHU with a list of HDU, the
   #            extensions. In AstroData, the extension numbering is zero-based.
   #            Eg. in IRAF myMEF[1] -> in AstroData ad[0]
   ad.insert(0, new_extension)
   ad.info()
   
   # Note that because the extension was named ('VAR',1) and that did not
   # conflict with any of the extensions already present, we did not have
   # to use auto_number=True.
   
   #   Here we insert the extension between the third and the fourth
   #   extensions.  Again, remember that the extension numbering is
   #   zero-based.
   ad.insert(3, new_extension, auto_number=True, do_deepcopy=True)
   
   # A ('VAR',1) extension already exists in ad, therefore auto_number must
   # be set to True.  Since we are insert the same new_extension, if we don't
   # deepcopy it, the EXTVER of the previous insert will also change.
   # Remember in Python, you might change the name of a variable, but both
   # will continue pointing to the same data: change one and the other will
   # change too.
   
   # Here we insert the extension between [SCI,3] and [SCI,4]
   # Note that the position we use for the index is ('SCI',4)
   # This is because we effectively asking for the new extension 
   # to push ('SCI',4) and take its place in the sequence.
   # 
   new_extension = adread['SCI',3]
   new_extension.rename_ext('VAR')
   ad.insert(('SCI',4), new_extension)
   ad.info()

   # Now that we have made a nice mess of ad, let's remove some extensions
   # Removing AstroData extension 4 (0-based array).
   ad.remove(4)
   ad.info()
   
   # Removing extension ['VAR',5]
   ad.remove(('VAR',5))
   ad.info()
   
   # Here is how to replace an extension.
   # Let's replace extension ('SCI',2) with the ('SCI',2) extension from adread.
   
   ##### .replace() is broken.  Will add example when it's fixed.
   
   
   # Finally, let's write this modified AstroData object to disk as a MEF file.
   # The input MEF was open in update mode.  If no file name is provide to the
   # write command, the file will be overwritten.  To write to a new file, 
   # specify a filename.
   ad.filename
   ad.write('newfile.fits')
   ad.filename
   
   # Note that any further write() would now write to 'newfile.fits' if no filename
   # is specified.

   # The pixel data and header data obviously can be accessed and modified.
   # More on pixel data manipulation in ???.  More on header manipulation in ???
   
   import numpy as np
   
   for extension in ad:
      # Obtain a numpy.ndarray. Then any ndarray operations are valid.
      data = ext.data
      type(data)
      np.average(data)
      
      # Obtain a pyfits header.
      hdr = ext.header
      print hdr.get('NAXIS2')
      
   # the numpy.ndarray can also be extracted this way.
   data = ad[('SCI',1)].data

   # To close an AstroData object.  It is recommended to properly close the object
   # when it will no longer be used.
   ad.close()
   adread.close()


Create New MEF Files
====================

The method ``write`` is use to write to disk a new MEF file from an AstroData 
object.  Here we show two ways to build that new AstroData object and create
a MEF file, in memory or on disk, from that AstroData object.

Create New Copy of MEF Files
----------------------------

Let us consider the case where you already have a MEF file on disk and you want
to work on it and write the modified MEF to a new file.

Here we open a file, make a copy, and write a new MEF file on disk::

  from astrodata import AstroData
  
  ad = AstroData('N20110313S0188.fits')
  ad.write('newfile2.fits')
  ad.close()

Since in Python and when working with AstroData objects, the memory can be
shared between variables, it is sometimes necessary to create a "true" copy
of an AstroData object to keep us from modifying the original.
  
By using ``deepcopy`` on an AstroData object the copy is a true copy, it has 
its own memory allocation.  This allows one to modify the copy while leave the 
original AstroData intact.  This feature is useful when an operation requires 
both the modified and the original AstroData object since by design a simple 
copy still point to the same location in memory. ::

   from astrodata import AstroData
   from copy import deepcopy
   
   ad = AstroData('N20110313S0188.fits')
   adcopy = deepcopy(ad)

In the example above, ``adcopy`` is now completely independent of ``ad``. 
This also means that you have doubled the memory footprint.  


Create New MEF Files from Scratch
---------------------------------

Another use case is creating a new MEF files when none existed before. The
pixel data needs to be created as a numpy ndarray.  The header must be created
as pyfits header. :: 

   from astrodata import AstroData
   import pyfits as pf
   import numpy as np
   
   # Create an empty header.  AstroData will take care of adding the minimal
   # set of header cards to make the file FITS compliant.
   new_header = pf.Header()
   
   # Create a pixel data array.  Fill it with whatever values you need.
   # Here we just create a fill gradient.
   new_data = numpy.linspace(0., 1000., 2048*1024).reshape(2048,1024)
   
   # Create an AstroData object and give it a filename
   new_ad = AstroData(data=new_data, header=new_header)
   new_ad.filename = 'gradient.fits'
   
   # Write the file to disk and close
   new_ad.write()
   new_ad.close()

