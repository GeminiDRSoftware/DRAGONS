
from astropy.io import fits
import hashlib
"""
This hndles "embedding" files (eg PDFs) in the ad object so that they can be
stored in the FITS file. We create a binary table attribute (tablename, which
defaults to .FIGURES), which has columns for filename, Content-Type (ie mime 
type), size, md5, and content, the latter of which is a varibale length binary 
field which contains the actual contents of the file.

This is useful for storing eg pdf figures generated during reduction inside
the reduced data fits file, which is especially useful when the fits file is
stored in the archive.

This code is not especially memory efficient - multiple copies of the data
may end up in memory. It is not intended for use with large files and will
likely need refactoring if that use cases arises.
"""

def embed_file(ad, filename, contenttype, data=None, tablename='FIGURES'):
    """
    Embed a file in an astrodata instance.
    :param ad: The ad instance to embed the file in
    :param filename: Filename of the file to embed
    :param contenttype: MIME type of the file
    :param data: bytes or bytesarray of the actual binary file contents, or a
    file-list object from which to read it from.
    If NULL, filename will be opened and read
    :param tablename: Name of the binary table attribute to store the file in.
    Defaults to 'FIGURES'
    :return: None
    """

    # Some convenience values are supported for contenttype:
    contenttype = 'application/pdf' if contenttype.upper() == 'PDF' else contenttype
    contenttype = 'image/png' if contenttype.upper() == 'PNG' else contenttype
    contenttype = 'image/jpeg' if contenttype.upper() in ('JPG', 'JPEG') else contenttype

    # If tablename already exists, read the columns into arrays
    thdu = getattr(ad, tablename, None)
    if thdu is not None:
        filenames = thdu['filename'].tolist()
        contenttypes = thdu['Content-Type'].tolist()
        sizes = thdu['size'].tolist()
        md5s = thdu['md5'].tolist()
        contents = thdu['content'].tolist()
    else:
        # Initialize empty column arrays
        filenames = []
        contenttypes = []
        sizes = []
        md5s = []
        contents = []

    # Append the new file to the arrays
    filenames.append(filename)
    contenttypes.append(contenttype)

    # If data None, read from the filename.
    if data is None:
        with open(filename, 'rb') as fp:
            content = bytearray(fp.read())
    # If data is a file-like, read from it
    elif hasattr(data, 'read') and callable(data.read):
        print("file like")
        content = bytearray(data.read())
        print(type(content))
    elif isinstance(data, bytes):
        print("bytes")
        content = bytearray(data)
    elif isinstance(data, bytearray):
        print("bytearray")
        content = data
    else:
        raise RuntimeError("Unknown data type in embed_file")

    contents.append(content)
    sizes.append(len(content))

    h = hashlib.new('md5')
    h.update(bytes(content))
    md5s.append(h.hexdigest())

    # Make a new table with the new arrays
    col1 = fits.Column(name='filename', format='128A', array=filenames)
    col2 = fits.Column(name='Content-Type', format='32A', array=contenttypes)
    col3 = fits.Column(name='size', format='J', array=sizes)
    col4 = fits.Column(name='md5', format='32A', array=md5s)
    col5 = fits.Column(name='content', format='PB()',
                       array=contents)
    hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5])
    setattr(ad, tablename, hdu)

def list_files(ad, fullinfo=False, tablename='FIGURES'):
    """
    List the files embedded in the astrodata instance
    :param ad: the astrodata instance with files embedded
    :param fullinfo: If False (default) return a list of filenames.
    If True, return a list of dicts containing 'filename', 'size', 'Content-Type' and 'md5' keys
    :param tablename: Name of the binary table attribute to store the file in.
    :return: list or dict as above
    """

    hdu = getattr(ad, tablename)
    retary = []

    for row in hdu:
        if fullinfo:
            d = {}
            for item in ('filename', 'size', 'Content-Type', 'md5'):
                d[item] = row[item]
            retary.append(d)
        else:
            retary.append(row['filename'])
    return retary

def extract_files(ad, todisk=True, filename=None, check=True, tablename='FIGURES'):
    """
    Extract files embedded in an astrodata instance
    :param ad: astrodata instance to extract from
    :param todisk: If True (default) write files to disk in current directory,
    using the filenames specified when they were embedded. If False, return
    a bytes instance with the content of the file.
    :param filename: If there are multiple files embedded and you want only
    one of them, give the filename here. If there are multiple files embedded
    and you do not specify a filename and you specify todisk=False, an exception
    will be raised
    :param check: If True (default) raise an exception if the size or md5
    does not match the content of the file.
    :return: if todisk=True, returns a list of files writted to disk.
    If todisk=False, returns the binary file content
    """
    hdu = getattr(ad, tablename)
    retary = []
    if len(hdu) > 1 and todisk is False and filename is None:
        raise RuntimeError("Cannot extract multiple files not to disk")
    for row in hdu:
        if filename is None or row['filename'] == filename:
            if check:
                if len(row['content']) != row['size']:
                    raise RuntimeError(f"File size mismatch for embedded file {row['filename']}")
                h = hashlib.new('md5')
                h.update(bytes(row['content']))
                if h.hexdigest() != row['md5']:
                    raise RuntimeError(f"MD5 mismatch for embedded file {row['filename']}")
            if todisk:
                with open(row['filename'], 'wb') as fp:
                    fp.write(bytes(row['content']))
                    retary.append(row['filename'])
            else:
                return bytes(row['content'])

    return retary