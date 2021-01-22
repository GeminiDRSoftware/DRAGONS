
.. _pull-request:  https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests

******************
Continuous Testing
******************

DRAGONS uses external tools that continuously perform tests after each commit in
a branch or in a Pull-Request_.

The Commit Status might be represented by a green tick on the right of the commit
message that indicates that all the checks passed, a yellow circle that means
that the checks are still running, or a red cross that tells us that
one or more checks failed. The Pull-Request status gives us this information but
it also informs us if there are any conflict between the origin branch and the
destination branch.

`GitHub Actions`_ is the first CI option that runs most of the tests that can be
visible by the public in general.

For tests that might use proprietary data and/or might contain sensitive
information relies on `Jenkins`_, which is only accessible by the
NOIR's Lab / GEMINI Staff.



GitHub Actions
==============

- What is GitHub Actions?



Jenkins
=======

