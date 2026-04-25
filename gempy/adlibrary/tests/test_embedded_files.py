import os

import astrodata
from gempy.adlibrary.embedded_files import embed_file, list_files, extract_files

def test_embed_bytes():
    ad = astrodata.AstroData()
    assert hasattr(ad, 'FIGURES') is False

    embed_file(ad, 'test.txt', contenttype='text/plain', data=b'abc')

    assert hasattr(ad, 'FIGURES') is True
    assert ad.FIGURES[0]['filename'] == 'test.txt'
    assert ad.FIGURES[0]['Content-Type'] == 'text/plain'
    assert ad.FIGURES[0]['size'] == 3
    assert len(ad.FIGURES[0]['content']) ==  ad.FIGURES[0]['size']
    assert ad.FIGURES[0]['md5'] =='900150983cd24fb0d6963f7d28e17f72'
    assert bytes(ad.FIGURES[0]['content']) == b'abc'

def test_tablename():
    ad = astrodata.AstroData()
    assert hasattr(ad, 'FOOBAR') is False

    embed_file(ad, 'test.txt', tablename='FOOBAR',
               contenttype='text/plain', data=b'abc')

    assert hasattr(ad, 'FOOBAR') is True
    assert ad.FOOBAR[0]['filename'] == 'test.txt'

def test_embed_fp(tmp_path):
    print("test embed fp")
    # Make a temporary file
    filename = tmp_path / 'file.dat'
    with open(filename, 'wb') as fp:
        fp.write(b"Hello, world")

    ad = astrodata.AstroData()

    # Open it again and embed the fp
    with open(filename, 'rb') as fp:
        embed_file(ad, 'file.dat', contenttype='text/plain', data=fp)

    assert hasattr(ad, 'FIGURES') is True
    assert ad.FIGURES[0]['filename'] == 'file.dat'
    assert ad.FIGURES[0]['Content-Type'] == 'text/plain'
    assert ad.FIGURES[0]['size'] == len("Hello, world")
    assert len(ad.FIGURES[0]['content']) == ad.FIGURES[0]['size']
    assert ad.FIGURES[0]['md5'] == 'bc6e6f16b8a077ef5fbc8d59d0b931b9'
    assert bytes(ad.FIGURES[0]['content']) == b'Hello, world'

def test_embed_from_file(tmp_path):
    # Make a temporary file
    filename = tmp_path / 'file.dat'
    with open(filename, 'wb') as fp:
        fp.write(b"Hello, world")

    ad = astrodata.AstroData()

    embed_file(ad, filename, contenttype='text/plain')

    assert hasattr(ad, 'FIGURES') is True
    assert ad.FIGURES[0]['filename'] == str(filename)
    assert ad.FIGURES[0]['filename'].endswith('file.dat')
    assert ad.FIGURES[0]['Content-Type'] == 'text/plain'
    assert ad.FIGURES[0]['size'] == len("Hello, world")
    assert len(ad.FIGURES[0]['content']) == ad.FIGURES[0]['size']
    assert ad.FIGURES[0]['md5'] == 'bc6e6f16b8a077ef5fbc8d59d0b931b9'
    assert bytes(ad.FIGURES[0]['content']) == b'Hello, world'

def test_list_files():
    ad = astrodata.AstroData()
    embed_file(ad, 'a.txt', contenttype='text/plain', data=b'abc')
    embed_file(ad, 'b.txt', contenttype='text/plain', data=b'defg')
    embed_file(ad, 'c.txt', contenttype='text/plain', data=b'hijkl')

    assert list_files(ad) == ['a.txt', 'b.txt', 'c.txt']

    lod = list_files(ad, fullinfo=True)
    a = lod[0]
    b = lod[1]
    c = lod[2]

    assert a['filename'] == 'a.txt'
    assert a['size'] == 3
    assert a['md5'] == '900150983cd24fb0d6963f7d28e17f72'
    assert a['Content-Type'] == 'text/plain'
    assert b['filename'] == 'b.txt'
    assert b['size'] == 4
    assert c['filename'] == 'c.txt'
    assert c['size'] == 5

def test_extract_files(tmp_path):
    ad = astrodata.AstroData()
    embed_file(ad, 'a.txt', contenttype='text/plain', data=b'abc')
    embed_file(ad, 'b.txt', contenttype='text/plain', data=b'defg')
    embed_file(ad, 'c.txt', contenttype='text/plain', data=b'hijkl')

    os.chdir(tmp_path)

    extract_files(ad)

    ld = os.listdir('.')
    assert 'a.txt' in ld
    assert 'b.txt' in ld
    assert 'c.txt' in ld

    with open('b.txt', 'rb') as fp:
        b = fp.read()

    assert b == b'defg'