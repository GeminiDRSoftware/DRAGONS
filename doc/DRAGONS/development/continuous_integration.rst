.. _continuous_integration:

Continuous Integration
======================

DRAGONS uses a Continuous Integration pipeline from commit to merge, in order
to:

1. Maintain basic coding standards across our software and packages.
2. Provide peace of mind to new and regular contributors.
3. Catch classes on common bugs and antipatterns early in the development
   process.
4. Perform testing and code analysis for feedback on complexity, coverage,
   performance, and integrity.

When using our :ref:`development environments <development_environments>` and
our repository, all CI will be automatically managed for you.

Linting, formatting, and quick checks (``pre-commit``)
------------------------------------------------------

.. ATTENTION::

   You can run these checks manually at any time by running

   .. code-block:: bash

      nox -s lint


    It performs exactly the same thing as:

    .. code-block:: bash

      pre-commit run --all



We perform quick conformance operations---specifically, formatting, linting,
and validating changes---when you try to commit your changes.

.. _pre-commit_docs: https://pre-commit.com/

The following tools are used to accomplish this through :ref:`pre-commit
<pre-commit_docs>`_. It runs the following other tools (with *no need to install
them yourself*):

.. _ruff_docs: https://docs.astral.sh/ruff/
.. _black_docs: https://black.readthedocs.io/en/stable/
.. _pre_commit_default_hooks: https://github.com/pre-commit/pre-commit-hooks?tab=readme-ov-file#pre-commit-hooks

+----------------------------------+--------------------------+
| Tool                             |                          |
+----------------------------------+--------------------------+
| ``black`` (`link <black_docs>`_) | Automated formatting     |
+----------------------------------+--------------------------+
| ``ruff`` (`link <ruff_docs>`_)   | Linting                  |
+----------------------------------+--------------------------+
| ``pre-commit-hooks``             | Miscellaneous validation |
+----------------------------------+--------------------------+

.. _noirlab_python_template: https://github.com/teald/python-standard-template

These tools will comprise a superset of NOIRLab's Python Coding Standard
outlined in the `Python standard template <noirlab_python_template>`.

Formatting
^^^^^^^^^^

DRAGONS uses the `black <black_docs>` formatter for all formatting in the
repository. We chose it for it focus on:

+ Consistency
+ Minimal diffs between changes
+ Clarity

You can learn more about the formatter in `the black documentation
<black_docs>`, but there are a few concepts that may be useful to your
development.

Trailing commas
***************

Trailing commas are used to expand collections. For example, if I omit the
trailing comma, the formatter will not change the following line:

.. code-block:: python

   def my_func_with_a_few_arguments(arg_1, arg_2, *, kwarg_1):
    """It has arguments, but does nothing."""

If we instead add a trailing comma after ``kwarg_``, it will expand the
arguments when the formatter is run:

.. code-block:: python

   def my_func_with_a_few_arguments(arg_1, arg_2, *, kwarg_1,):
    """It has arguments, but does nothing."""

becomes

.. code-block:: python

  def my_func_with_a_few_arguments(
      arg_1,
      arg_2,
      *,
      kwarg_1,
  ):
      """It has arguments, but does nothing."""

This also happens if the arguments become *too large to fit on one line,
alone*. black will automatically add the trailing comma.

*Why should you use this?* It is not just difficult to parse functions with
many arguments on a single line; it makes the diffs between code versions much
less clear than they otherwise would be. If one argument is modified, a whole
line of arguments is changed and history about prior changes to arguments is
obfuscated in the commit history.

If a function's arguments, or the contents of a literal ``dict``, ``list``, or
other collection, are numerous and not obvious by eye, if can be a good idea to
just add the trialing comma yourself..

Ignoring the formatter
**********************

To ignore lines of code when formatting, you add ``# fmt: off`` and ``# fmt:
on`` before and after (respectively) the lines to be ignored. For example:

.. code-block:: python

   # The below code will be formatted
   code = ("my code with unnecessary parentheses")

   # The code after the below comment will not.
   # fmt: off
   my_matrix = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]
   ]

   # fmt: on
   # After the above comment, formatting is applied.
