.. issues_and_limitations.rst

.. _issues_and_limitations:

**********************
Issues and Limitations
**********************

.. _double_messaging:

Double messaging issue
======================
If you run ``Reduce`` without setting up a logger, you will notice that the
output messages appear twice.  To prevent this behaviour set up a logger.
This will send one of the output stream to a file, keeping the other on the
screen.  We recommend using the DRAGONS logger located in the
``logutils`` module and its ``config()`` function:


.. code-block:: python
    :linenos:

    from gempy.utils import logutils
    logutils.config(file_name='gmosls_tutorial.log')
