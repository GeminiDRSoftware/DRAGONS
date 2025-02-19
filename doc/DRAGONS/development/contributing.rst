.. _contributing_docs_page:

Contributing to DRAGONS
=======================

If you're new here, thanks for taking the time to contribute to the DRAGONS
project! This guide is for how to contribute to the DRAGONS repository.

This page outlines the process from fork/branch to merging into the primary
branch on the DRAGONS repository.

|image0|

.. |image0| image:: https://mermaid.ink/img/pako:eNpVkMtqwzAQRX9lmLXzA6YE_Eh3hZBmFcuLqTSRTGzJ1SOhhPx7ZYeWdjdzz4F53FE6xVjieXQ3achHOLbCAkDVvTp_Aeeh9mSl6WGz2ULd7dM4woE_E4fYr2a9kqarUnQTRVYvH357zDj85W3X5EkLOvB14NuTNZnBnkJgtVq7NW2XtJKS5_gv33WNIas5wMRes-qxwFxNNKh8wn1xBEbDEwssczkO2kSBwj6ySHm99y8rsYw-cYHeJW2wPNMYcpdmlVdvB9Kept-U1RCdf3u-aP1UgTPZk3M_zuMbubhnNA?type=png
   :target: https://mermaid.live/edit#pako:eNpVkMtqwzAQRX9lmLXzA6YE_Eh3hZBmFcuLqTSRTGzJ1SOhhPx7ZYeWdjdzz4F53FE6xVjieXQ3achHOLbCAkDVvTp_Aeeh9mSl6WGz2ULd7dM4woE_E4fYr2a9kqarUnQTRVYvH357zDj85W3X5EkLOvB14NuTNZnBnkJgtVq7NW2XtJKS5_gv33WNIas5wMRes-qxwFxNNKh8wn1xBEbDEwssczkO2kSBwj6ySHm99y8rsYw-cYHeJW2wPNMYcpdmlVdvB9Kept-U1RCdf3u-aP1UgTPZk3M_zuMbubhnNA

Fork the repository
-------------------

.. admonition:: DRAGONS core developers

  This step is only necessary if you do not have access to the DRAGONS
  repository to make branches. 

  If you do have access, you can continue to the next section.


.. _github_contributing_with_forks_link: https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project

In order to make changes, you'll need a copy of the repository you can freely
edit. `Github has a page detailing this process from start to finish
<github_contributing_with_forks_link_>`_.

Once you've made your changes and are ready to include them in the DRAGONS
repo, move on to the next step.

Making a Pull Request
---------------------

Whether using a fork or branch, all contributions should become a Pull Request
(PR) once you're ready to contribute it to the repository.

.. _github_draft_pr_docs: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request#creating-the-pull-request

.. admonition:: Draft PRs

   You can also make a draft PR to run the CI/CD pipeline while signalling your
   contribution is not ready for review. See `Creating a Pull Request
   <github_draft_pr_docs_>`_.


.. _github_getting_started_prs: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes
.. _attlassian_pr_guide: https://www.atlassian.com/blog/git/written-unwritten-guide-pull-requests

If you're new to PRs, you may find the following guides and best practices
helpful:

* `Getting started with pull requests (GitHub) <github_getting_started_prs_>`_
* `The (written) unwritten guide to pull requests (Atlassian) <attlassian_pr_guide_>`_


Code Review
-----------

Once a PR has been made and marked ready for review, you will be assigned
one or more core developers to review your code. Depending on the scope and
content of your PR, as well as the backlog of work on the DRAGONS team's end,
the time this takes will vary.

Best Practices
^^^^^^^^^^^^^^

The code review process should be a collaborative and positive experience for
both parties, even if the code reviewed is eventually rejected.

All contributions, discussions, and code reviews are subject to our :ref:`Code of
Conduct <code_of_conduct>`.

In general try to...

* **Stay positive**: We're working to improve software, and it's unlikely
  anyone is trying to be malicious. Feedback may be succinct/direct, but avoid
  accusatory ("you did this wrong", "your mistake") or condescending
  ("obviously", "trivial", "sloppy") language. Your point can be communicated
  without it, and their inclusion provide neither insight nor value!
* **Be specific**: Communicate with example code where possible; don't write a
  paragraph when a code snippet could communicate the same idea. Focus on the
  scope of the PR, and try not to go beyond it unless really necessary.
* **Stay constructive**: Blanket responses, particularly blanket rejections of
  some or all of a PR, are not helpful. If you think there's a problem, explain
  why you think there's a problem. If you find yourself reacting significantly
  to a PR, e.g., you start to get frustrated, step away. You can also ask for
  another person to help!
* **Keep responses as long as they need to be**: If you find yourself needing to spend
  a lot of time reviewing a small PR, or one part of a larger PR, take a step
  back and think about whether this could be an issue with implementation, the
  PR, or documentation. Is there a better way to frame what you're trying to
  say, through code, text formatting (e.g., a list) or something else?
* **Nitpick responsibly**: Things like code style and naming conventions will
  be enforced in CI/CD. If you have other nitpicky comments, feel free to leave
  them and prefix them with "``nitpick``" to make it 100% clear they are not
  critical. **Do not assume your tone is sufficient in communicating this!**

Reviewer Checklist
^^^^^^^^^^^^^^^^^^

Below is a checklist for reviewers of code, to help keep in mind specific
topics. Use it to guide your PR, but don't hesitate to go beyond it if you find
something important.

You can copy/paste this checklist directly into the PR. It uses markdown
formatting.

.. ========= READ THIS EDITORS =========
.. If you change anything below that isn't a typo, please bump the version up!

.. code-block:: markdown

   This is the DRAGONS code review template (version 0).

   ## Instructions

   For some PRs, some of the points below will not apply. Just check them off
   as you come to them for completeness and consistency. Checking off an item
   is intended to communicate "I have considered this bullet point and believe
   it's irrelevant or that my concerns, if any, have been addressed."

   ## Checklist

   - [ ] Functionality

     - Do changes address all functionality outlined in the PR?
     - Are changes maintainable?
     - Do changes conflict or couple with other components of DRAGONS?

   - [ ] Testing

     - Have edge-cases been appropriately considered?
     - Do test names describe what the test is doing?
     - Do tests, within reason, follow DRY (Don't Repeat Yourself) principles?
     - Can new tests be run on your machine without any setup?

   - [ ] Readability

     - Are the changes straightforward to understand?
     - Do changes significantly increase the complexity of the code?
     - Are variables/functions/classes named descriptively?

   - [ ] Usability

     - Are there any side-effects that are not obvious to a user?
     - Are exceptions appropriately handled, and not discarded?
     - Has relevant documentation been created or updated?

   - [ ] Big Picture (beyond this PR/review: create a new issue/ticket!)

     - Are there opportunities for automation?
     - Does this PR beget other features?
     - Did you have any ideas for improvements or features outside the scope of
       this PR?

.. _roadmap_code_review_best_practices: https://roadmap.sh/best-practices/code-review
.. _code_review_pyramid: https://www.morling.dev/blog/the-code-review-pyramid/

This should help guide your review. If it's your first time reviewing/being
reviewed, or you feel stuck, check out these resource graphics:

* `Code review best practices (roadmap.sh) <roadmap_code_review_best_practices_>`_
* `The Code Review Pyramid (Gunnar Morling) <code_review_pyramid_>`_
