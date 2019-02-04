
.. _`AstroConda`: https://astroconda.readthedocs.io/en/latest/
.. _`DRAGONS`: https://github.com/GeminiDRSoftware/DRAGONS


.. _get_ready:

Get Ready
=========

`DRAGONS`_ requires several libraries that could be installed individually but
you can have all of them by `installing AstroConda <https://astroconda.readthedocs.io/en/latest/getting_started.html#getting-started-jump>`_.
Just click the link and follow the guidelines. It should take no more than ten
minutes (if your internet is fast).

Once you have `AstroConda`_ installed and you have set your new Virtual
Environment, you can download `DRAGONS`_ and install it.

You can find the last release version in the link below:

https://github.com/GeminiDRSoftware/DRAGONS/archive/v2.0.10.zip

Decompress the downloaded file and access it using a (bash) terminal. Make sure
your Virtual Environment is activated. Let's perform some checks before you
install DRAGONS.

.. code-block:: none

    $ cd $PATH_TO_DRAGONS
    $ which python
    /path/to/your/python

When you type `which python`, you should see the full path to your python
binary file and it should be living inside your Virtual Environment folder.
Before actually installing, you must build a library using Cython:

.. code-block:: none

    $ cythonize -a -i gempy/library/cyclip.pyx
    running build_ext

No other messages should appear. After this, test if everything is fine:

.. code-block:: none

    $ python setup.py test
    (...)

You will see a lot of messages. If you find any error, please, contact us. If
not, you can proceed:

.. code-block:: none

    $ python setup.py install

or

.. code-block:: none

    $ pip install .

We recommend installing DRAGONS with `pip` because it makes it easier to perform
updates on your machine.
