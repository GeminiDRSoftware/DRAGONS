
import os


"""This file contains the following utilities:
    deleteFile (filename)
    copyFile (input, output)
"""


def deleteFile (filename):
    """Delete a file.

    The file 'filename' will be deleted if it exists, even if it does not
    have write access.

    @param filename: the name of a file
    @type filename: string
    """

    if os.access (filename, os.F_OK):
        os.remove (filename)


def copyFile (input, output):
    """Copy the text file 'input' to the file 'output'."""

    ifd = open (input)
    lines = ifd.readlines()
    ifd.close()
    ofd = open (output, "w")
    ofd.writelines (lines)
    ofd.close()
