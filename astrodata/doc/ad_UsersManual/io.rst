.. io:

.. _io:

*******************************************************
MEF Input/Output Operations and Extensions Manipulation
*******************************************************

In this section, we will show and discuss how to read and write to a 
Multi-Extension FITS (MEF) file.

A MEF file is a Primary Header Unit (PHU) with a list of Header Data Units
(HDU), commonly referred to as extensions.  In AstroData, the extension
numbering is zero-indexed, the first extension is ``myAstroData[0]`` where 
``myAstroData`` is an open AstroData object.  If you are familiar with
IRAF, then keep in mind that in IRAF the extensions are 1-indexed.  For
example::

   # In IRAF
   display myMEF[1]
   
   # With AstroData
   numdisplay.display(myAstroData[0].data) 

The first data extension is 1 in IRAF, but it is 0 in AstroData.  All Python
arrays are zero-indexed, so AstroData was made compliant with modern practice.


**Try it yourself**


If you wish to follow along and try the commands yourself, download
the data package, go to the ``playground`` directory and copy over
the necessary files.

::

   cd <path>/gemini_python_datapkg-X1/playground
   cp ../data_for_ad_user_manual/N20110313S0188.fits .
   cp ../data_for_ad_user_manual/N20110316S0321.fits .
   cp ../data_for_ad_user_manual/N20111124S0203.fits .

Then launch the Python shell::

   python


.. highlight:: python
   :linenothreshold: 5


Open and access existing MEF files
==================================

An AstroData object can be created from the name of the file on disk, a URL,
or from PyFITS HDUList or HDU instance.  An existing MEF file can be opened as 
an AstroData object in ``readonly``, ``update``, or ``append`` mode.  
The default is ``readonly``.

Here is a very simple example on how to open a file in ``readonly`` mode, 
check the structure, and then close it::

   from astrodata import AstroData
   
   ad = AstroData('N20111124S0203.fits')
   ad.info()
   ad.close()

The first line, imports the ``AstroData`` class.  The ``info`` method prints
to screen the list of extensions, their name, size, data type, etc.  Here's
what the example above will output::

   Filename: N20111124S0203.fits
       Type: AstroData
       Mode: readonly
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       179
           phu.header    Header
   [0]     ('SCI', 1)    ImageHDU      1        71    (4608, 1056)  float32
              .header    Header
              .data      ndarray
   [1]     ('SCI', 2)    ImageHDU      2        71    (4608, 1056)  float32
              .header    Header
              .data      ndarray
   [2]     ('SCI', 3)    ImageHDU      3        71    (4608, 1056)  float32
              .header    Header
              .data      ndarray
   [3]     ('SCI', 4)    ImageHDU      4        71    (4608, 1056)  float32
              .header    Header
              .data      ndarray
   [4]     ('SCI', 5)    ImageHDU      5        71    (4608, 1056)  float32
              .header    Header
              .data      ndarray
   [5]     ('SCI', 6)    ImageHDU      6        71    (4608, 1056)  float32
              .header    Header
              .data      ndarray


To open the file in a mode other than ``readonly``, one specifies the setting 
of the ``mode`` argument::

   ad = AstroData('N20111124S0203.fits', mode='update')


Accessing the content of a MEF file
-----------------------------------

Conceivably one opens a file to access its content. Manipulations of the 
pixel data and of the headers are covered in more details in later sections 
(:ref:`Section 5 - Pixel Data <data>` and 
:ref:`Section 4 - FITS Headers <headers>`, respectively). Here we show 
a few very basic examples on how to access pixels and headers.

The pixel data in an AstroData object are stored as Numpy ``ndarray``.
Any ``ndarray`` operations are valid.  We use Numpy's ``average`` function
in the example below.

The headers in an AstroData object are stored as PyFITS headers.  Any
PyFITS header operations are valid.

::

   from astrodata import AstroData
   import numpy as np
   
   ad = AstroData('N20111124S0203.fits')
   
   for extension in ad:      
      print 'Extension :', extension.extname(), ',', extension.extver()
      #
      # Access the pixel data
      data = extension.data  
      print 'data is of : ', type(data)    
      print 'The pixel data type is: ', data.dtype
      print 'The average of all the pixels is: ', np.average(data)
      #
      # Access the header
      hdr = extension.header
      print 'The value of NAXIS2 is: ', hdr.get('NAXIS2')
      print
   
*(Python Beginner's Note 1: The ``#`` on line 8 and 14 are not necessary to the 
code but simplify
the cut and paste of the statements from the HTML page to the Python
shell, without affecting readability.)*

*(Python Beginner's Note 2: In the Python shell, when you are done inputing the 
statements of a loop,
you indicate so by typing return to create an empty line.  So, after you
have written the last ``print`` statement, type return on the ``...`` line,
this will launch the execution of the loop.)*

Now let us discuss the example.

As stated above, the pixel data are stored in ``numpy.ndarray`` objects.  
Therefore, Numpy needs to be imported if any ``numpy`` operations is to
be run on the data.  This is done on Line 2, using the standard import 
convention for Numpy.

On Line 6, the for-loop that will access the extension sequentially is defined.
Only the extensions are returned, the Primary Header Unit (PHU) is not
sent to the loop.  Access to the PHU is discussed in 
:ref:`Section 4 - FITS Headers <headers>`.

In an AstroData object, each extension is given, in memory if not on disk, 
an extension name and an extension version.  Line 7 accesses that information.

On lines 10 to 13, the pixel data for the current extension is assigned to the
variable ``data``, and then the array is explored a bit.

On lines 16 and 17, the header associated with the extension is assigned to
the variable ``hdr``, and the value for the keyword ``NAXIS2`` is retrieved.

Note that for both ``data`` and ``hdr``, the pixels or the headers are 
NOT copied, the new variables simply point to the information stored in the
AstroData object.  If ``data`` or ``hdr`` are modified, the AstroData object
itself will be modified.

In the example above, a loop through the extensions is used.  To access a specific
extension by name, it is also possible to do something like this::
      
   data = ad['SCI',1].data
   print 'Value of NAXIS2: ', ad['SCI',1].header.get('NAXIS2')

or if not using the names, using the positional number for the extension::

   header = ad[0].header
   print 'Extension name and version for extension 0: ', \
      ad[0].extname(), ad[0].extver()

Note that the extension positions are zero-indexed, ``ad[0]`` is not the
PHU, it is the first extension.
  

Modify Existing MEF Files
=========================
To modify an existing MEF file, it must have been opened in the ``update``
mode.  While a MEF opened in any mode can be modified at will in memory,
only an file opened in ``update`` (or ``append``) mode can be overwritten 
on disk. 

Here we give examples on how to append an extension, how to insert an 
extension, and how to remove an extension.  Finally we show 
how to write the updated AstroData object to back disk as a MEF file.

Extension manipulations have been chosen for this discussion, but any other
type of modifications, eg. pixel arithmetics, header editing, etc. could
have been chosen instead.  Extension manipulations are a good introduction
to the structure and extension-naming convention of the AstroData MEF file
representation.


Opening the files
-----------------

For the extension manipulation examples of the next subsections, two files
are needed, one to serve as the main file to edit, and another from which
we can extract extension from for inserting or appending into the first.

::

   from astrodata import AstroData
   
   ad = AstroData('N20110313S0188.fits', mode='update')
   ad.info()
   
   adread = AstroData('N20110316S0321.fits', mode='readonly')
   new_extension = adread["SCI",2]
   new_extension.info()

The first step is always to import the ``AstroData`` class (line 1).  
On Line 3, the main dataset is open as an AstroData object in ``update``
mode.  The other dataset is open in ``readonly`` mode (line 6).  Note that
the ``readonly`` is optional as this is the default mode.

An extension is "extracted" from the second dataset on line 7.  It is 
important to realize that this does NOT create a copy of the extension.
The variable ``new_extension`` simply *points* to the data stored in ``adread``.

The first dataset's structure is (from line 4, ``ad.info()``)::

   Filename: N20110313S0188.fits
       Type: AstroData
       Mode: readonly
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       179
           phu.header    Header
   [0]     ('SCI', 1)    ImageHDU      1        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [1]     ('SCI', 2)    ImageHDU      2        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [2]     ('SCI', 3)    ImageHDU      3        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray


The structure of the new extension is (from line 8. ``new_extension.info()``)::

   Filename: N20110316S0321.fits
       Type: AstroData
       Mode: update
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       147
           phu.header    Header
   [0]     ('SCI', 2)    ImageHDU      1        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray


Appending an extension
----------------------

Appending and inserting extension can be tricky.  The first difficulty comes
from the naming of the extensions.  No two extensions are allowed to have
the same EXTNAME, EXTVER combination.  When appending or inserting, the
user must either specify the EXTNAME, EXTVER of the new extension or use
the ``auto_number`` options which tries to do something sensible to keep the
MEF structurally valid.

The second difficulty is due to the fact an assignment of an extension to
a variable is 
just a "reference", a "link", it is not a copy.  Any changes to the
variable standing as a pointer to an extension in an AstroData object will
affect the AstroData object itself.

The ``append`` method adds an extension as the end of a dataset. Here is an 
example appending an extension to a dataset.  Further discussion follows.

::

   ad.append(new_extension, auto_number=True, do_deepcopy=True)
   ad.info()

The AstroData method ``append()`` is used to append an extension to
an AstroData object.  On the first line, the extension ``new_extension``
gets appended to the ``ad`` dataset.  

As you see in the previous subsection, the extension name and extension
version number of the extension in ``new_extension`` is ['SCI', 2].  There
is already an extension named and versioned that way in ``ad`` (see the
result of ``ad.info()`` in the previous subsection).  Therefore, to avoid
conflict, the argument ``auto_number`` is set to ``True``.

As you can see from the output of ``ad.info()`` after the ``append`` call::

   Filename: N20110313S0188.fits
       Type: AstroData
       Mode: update
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       179
           phu.header    Header
   [0]     ('SCI', 1)    ImageHDU      1        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [1]     ('SCI', 2)    ImageHDU      2        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [2]     ('SCI', 3)    ImageHDU      3        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [3]     ('SCI', 4)    ImageHDU      4        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray


There is a new extension at the bottom of the list named ['SCI', 4] (lines 
18-20).  The ``auto_number`` feature figured out that ['SCI', 1 to 3] already 
existed and assigned the new ['SCI'] extension a version number of 4 to avoid 
conflict.

Now, there is a second issue to deal with.  Since ``new_extension`` is just
a reference to the extension stored in ``adread``, when ``auto_number`` 
changes the version number to 4, the extension in ``adread`` will also be
modified, corrupting the source.  Moreover, if other modifications are made
to the extension inserted in ``ad``, it will modify ``adread`` too. There are 
times when it will not matter at all, for example if ``adread`` is scheduled to be closed anyway, 
but if ``adread`` is to be used later in the script, it should not be modified 
like that.  

To cut the link between the extension appended to the dataset and the
source, the argument ``do_deepcopy`` is set to ``True``.  This will often be
needed.  It is not the default because of memory usage concerns, and to force
the user to think before creating copies and decide if it is truly needed.


Inserting an extension
----------------------

When inserting an extension into an AstroData object, the same caution to
extension name and version, and to the reference versus copy issues applies.
Instead of repeating ourselves, we refer to users to the discusssion above in
the "Appending" section.

With that in mind, let us present a few examples of insertion.

Simple insertion
^^^^^^^^^^^^^^^^

To insert an extension between the PHU and the first extension::

   new_extension = adread['SCI',1]
   new_extension.rename_ext('VAR')
   new_extension.info()

   ad.insert(0, new_extension)
   ad.info()

On Line 2, we rename the extension to 'VAR' simply to make it stand out
once inserted.  Also, note that since the new extension is named ['VAR',1],
it does not conflict with any of the extensions already present in ``ad``,
therefore there were no need for activating the ``auto_number`` option. We
did not use ``do_deepcopy`` either.  As consequence, the source of 
``new_extension``, ``adread``, has been modified (in memory). 

Before and after insertion, ``new_extension`` has this structure::

   Filename: N20110316S0321.fits
       Type: AstroData
       Mode: update
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       147
           phu.header    Header
   [0]     ('VAR', 1)    ImageHDU      1        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray

The actual insertion takes place on Line 5.  The syntax requires the 
first argument to be the position number, or the name and version,
of the extension to "push".  The new extension will be inserted *before*
the extension specified in the statement.  The second argument is obviously
the extension to insert into ``ad``.

After insertion, ``ad`` looks like this::

   Filename: N20110313S0188.fits
       Type: AstroData
       Mode: update
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       179
           phu.header    Header
   [0]     ('VAR', 1)    ImageHDU      1        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [1]     ('SCI', 1)    ImageHDU      2        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [2]     ('SCI', 2)    ImageHDU      3        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [3]     ('SCI', 3)    ImageHDU      4        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [4]     ('SCI', 4)    ImageHDU      5        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray

As one can see, ``new_extension`` is now the first extension in the file 
structure, or at position 0 (Lines 9-11), and the other extensions have been 
moved "down".

A tricky insertion
^^^^^^^^^^^^^^^^^^
The name of the new extension ['VAR',1] was not in conflict with any 
pre-existing extensions in the original AstroData object.  Let us insert
the new extension again, in another position in the current AstroData object.
This time, there is a ['VAR', 1] in the AstroData object, and ``auto_number``
and ``do_deepcopy`` are required.

The ``auto_number`` option is required to avoid the extension name clash.
Less obvious is why ``do_deepcopy`` is required, assuming that we do not
care about the impact on ``adread``.  The reason is subtle, and clearly 
illustrate why extension manipulations is probably the most tricky concept
in this manual, yet fortunately not that commonly needed.

When we inserted ``new_extension`` above, we did not use ``do_deepcopy``.
Therefore, if we were to modify ``new_extension``, like through ``auto_number``,
we would be modifying not only the source, ``adread``, but also that extension
we have already added to ``ad`` !  

As you can see, it is vitally important to understand was is a true copy and
what is a reference to something else when dealing with extensions.  Beginners
might want to use ``do_deepcopy=True`` as a default, until they are comfortable
with the concept of references.  *Beware* however that memory usage can
rise significantly.

Here is how one would insert ``new_extension`` somewhere else in ``ad``.
In the example, the extension is inserted between the current third and
fourth extension.  Since position ID are zero-indexed, this means between
position 2 and 3.
 
::
 
   ad.insert(3, new_extension, auto_number=True, do_deepcopy=True)
   ad.info()

Look at what happened to the name of the newly inserted extension 
(Lines 18-20)::

   Filename: N20110313S0188.fits
       Type: AstroData
       Mode: update
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       179
           phu.header    Header
   [0]     ('VAR', 1)    ImageHDU      1        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [1]     ('SCI', 1)    ImageHDU      2        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [2]     ('SCI', 2)    ImageHDU      3        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [3]     ('VAR', 5)    ImageHDU      4        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [4]     ('SCI', 3)    ImageHDU      5        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [5]     ('SCI', 4)    ImageHDU      6        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray

The automatic renumbering assigned an extension number of 5 to the newly
inserted extension.  One might have expected that 2 would be assigned as the
next available version number of the 'VAR' name.  This behavior was designed
to prevent the software from making the scientific assumption that the new
extension is in anyway associated with another.  Normally, it is assumed that
all extension with a given EXTVER are scientifically associated.  ``auto_number``
has no way to know which extension is scientifically associated with an other.
The purpose of ``auto_number`` is solely to keep the AstroData structure sound
and prevent corruption due to clashing name/version pairs.  It is the job
of the programmer, who has the scientific knowledge of the associations, to
name and version the extensions correctly when that matters. 
   
Using name/version pairs instead of position ID
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The position at which the insertion is to take place can be given as the 
positional ID like in the examples above, or by specify the extension name
and version.

::
   
   new_extension = adread['SCI',3]
   new_extension.rename_ext('VAR')
   ad.insert(('SCI',4), new_extension)
   ad.info()

In this example, a new extension is retrieved from the source and renamed
'VAR' to avoid name conflict and keep the example simple.  The insertion takes
place on the third line where the position of insertion is specfied as
``('SCI',4)``.  The resulting ``ad`` looks like this::

   Filename: N20110313S0188.fits
       Type: AstroData
       Mode: update
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       179
           phu.header    Header
   [0]     ('VAR', 1)    ImageHDU      1        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [1]     ('SCI', 1)    ImageHDU      2        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [2]     ('SCI', 2)    ImageHDU      3        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [3]     ('VAR', 5)    ImageHDU      4        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [4]     ('SCI', 3)    ImageHDU      5        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [5]     ('VAR', 3)    ImageHDU      6        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [6]     ('SCI', 4)    ImageHDU      7        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray


The new extension pushed ('SCI',4) down and took its place in the sequence.


Removing an extension
---------------------

Compared to appending and inserting extensions, removing them is a breeze.
As before, the extension to remove can be speficied with the position number 
or with the extension name and version.  Just remember that the position 
numbers are zero-indexed.

::

   ad.remove(4)
   ad.info()
   
   ad.remove(('VAR',5))
   ad.info()

After the two removal above, ``ad`` looks like this::

   Filename: N20110313S0188.fits
       Type: AstroData
       Mode: update
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       179
           phu.header    Header
   [0]     ('VAR', 1)    ImageHDU      1        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [1]     ('SCI', 1)    ImageHDU      2        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [2]     ('SCI', 2)    ImageHDU      3        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [3]     ('VAR', 3)    ImageHDU      4        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [4]     ('SCI', 4)    ImageHDU      5        39    (2304, 1056)  float32
              .header    Header
              .data      ndarray


Updating the existing file on disk
----------------------------------

If a file has been opened in ``update`` mode, the file on disk can be
overwritten with the ``write()`` command. ::
      
   ad.filename
   ad.write()

The first line will print the file name currently associated with the 
AstroData object, ``ad``.  This the file that will be written to.

More often though, the idea is to the write the modified output to a new
file.  This can be done regardless of the ``mode`` used when the file was 
opened.  All that is needed is to specify a new file name.  Note that
this will change the file name associated with the AstroData object, 
permanently, any other ``write`` commands will write to the new file name.

::
  
   ad.filename
   ad.write('newfile.fits')
   ad.filename
   
Before the write, the file name is ``N20110313S0188.fits``.  After the write,
the file name is ``newfile.fits``.


Closing and cleaning up
-----------------------

It is recommended to properly close the opened AstroData objects when they are 
no longer needed::

   ad.close()
   adread.close()

If you have been following along, the input file on disk was modified by
one of the ``write`` examples above.  We will need the unmodified file in
the next section.  To restore the file to the original::

   import shutil
   shutil.copy('../data_for_ad_user_manual/N20110313S0188.fits', '.')


Create New MEF Files
====================

A new MEF file can be created from a copy of an existing file or created
from scratch with AstroData objects. 


Create New Copy of MEF Files
----------------------------

Let us consider the case where you already have a MEF file on disk and you want
to work on it and write the modified MEF to a new file.

Basic example
^^^^^^^^^^^^^
Open a file, make modifications, write a new MEF file on disk::

   from astrodata import AstroData
   
   ad = AstroData('N20110313S0188.fits')
   ... modifications here ...
   ad.write('newfile2.fits')
   ad.close()

Needing true copies in memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Since in Python, and when working with AstroData objects, the memory can be
shared between variables, it is sometimes necessary to create a "true" copy
of an AstroData object to keep us from modifying the original.
  
By using ``deepcopy`` on an AstroData object the copy is a true copy, it gets 
given its own memory allocation.  This allows one to modify the copy while 
leaving the original AstroData intact.  This feature is useful when an 
operation requires both the modified and the original AstroData object since 
by design a simple copy or assignment points to a common location in memory. 
Use carefully however, your memory usage can grow rapidly if you over-use. ::

   from astrodata import AstroData
   from copy import deepcopy
   
   ad = AstroData('N20110313S0188.fits')
   adcopy = deepcopy(ad)

In the example above, ``adcopy`` is now completely independent copy of ``ad``. 
This also means that you have doubled the memory footprint.  Also note that
both copies have the same file name associated to them; be mindful of that
if you ``write`` the files back to disk.


Create New MEF Files from Scratch
---------------------------------

Another use case is creating a new MEF files or AstroData object when none 
existed before. The pixel data needs to be created as a Numpy ``ndarray``.  
The header must be created as PyFITS header. IMPORTANT: AstroData currently
is not compatible with ``astropy.io.fits``, one *must* use the standalone 
PyFITS module (it comes with Ureka).

:: 

   from astrodata import AstroData
   import pyfits as pf
   import numpy as np
   
   # Create an empty header.
   new_header = pf.Header()
   
   # Create a pixel data array.
   new_data = np.linspace(0., 1000., 2048*1024).reshape(2048,1024)
   
   # Create an AstroData object and give it a filename
   new_ad = AstroData(data=new_data, header=new_header)
   new_ad.filename = 'gradient.fits'
   
   # Write the file to disk and close
   new_ad.write()
   new_ad.close()

The input header does not need to have anything in it (Line 6).  In fact, if 
you are really creating from scratch, it is probably better to leave it empty 
and populate it after the creation of the AstroData object.  Upon creation,
AstroData, through PyFITS, will take care of adding the minimal set of header
cards to make the file FITS compliant.

The pixel data array must be a ``ndarray``.  On Line 9, we create a
1024 x 2048 array, filled with a gradient of pixel value.

It is important to attach a name to the AstroData object (line 13) if it is to 
be written to disk.  No default names are assigned to new AstroData objects.

