
Testing DRAGONS
===============

Testing Pipeline
----------------

The DRAGONS testing pipeline consists of:

+ Initial testing run using Github Actions
+ A more comprehensive test suite run using a managed Jenkins server

Ideally, changes to source code should be associated with tests and we aim for
our test coverage to increase or stay the same as source code is modified and
added. CodeCov is used to assess test coverage, and will post coverage updates
to pull requests.
