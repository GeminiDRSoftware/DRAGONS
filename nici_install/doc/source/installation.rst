
Installation
============

This is a preliminary release of the NICI Reduction package.

The NICI module comes as a tar file located in: 

  *ritchie:/spare/nici.tar*

1. Grab the tar file into your local directory.

2. Untar 

 ::

  tar xf nici.tar

3. Install the module:

 - If you can write into the python site-packages directory then type:

 ::       

       python setup.py install

 - Otherwise install the module in some other directory; e.g. /tmp.


 ::       

       python setup.py install --home=/tmp

       # Add this new directory to your PYTHONPATH. (mostly in your '.cshrc' file) 
       # Notice the extra 'lib/python' that python installation added.

       # (for cshell):

       setenv PYTHONPATH ${PYTHONPATH}:/tmp/lib/python

       # source .cshrc to update your unix environment.

4. Now you are ready to use the NICI reduction scripts.

