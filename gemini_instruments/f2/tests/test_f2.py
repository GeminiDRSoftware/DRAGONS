#!/usr/bin/env python

import glob
import os
import pytest

import astrodata
import gemini_instruments


@pytest.fixture
def f2_files(path_to_inputs):
    def get_files(instrument):
        return glob.glob(os.path.join(path_to_inputs, instrument, "*fits"))

    gemini_files = []
    gemini_files.extend(get_files("F2"))
    gemini_files.sort()

    return gemini_files


@pytest.mark.xfail(reason="AstroFaker changes the AstroData factory")
def test_is_right_type(f2_files):
    for _file in f2_files:
        ad = astrodata.open(_file)
        assert type(ad) == gemini_instruments.f2.adclass.AstroDataF2


def test_is_right_instance(f2_files):
    for _file in f2_files:
        ad = astrodata.open(_file)
        assert isinstance(ad, gemini_instruments.f2.adclass.AstroDataF2)


def test_extension_data_shape(f2_files):
    for _file in f2_files:
        ad = astrodata.open(_file)
        data = ad[0].data
        assert data.shape == (1, 2048, 2048)


# def test_tags(f2_files):
#     for _file in f2_files:
#         ad = astrodata.open(_file)
#         tags = ad.tags
#
#         expected_tags = {
#             'F2',
#             'SOUTH',
#             'GEMINI',
#         }
#
#         assert expected_tags.issubset(tags)
#
#
# def test_can_return_instrument(f2_files):
#     for _file in f2_files:
#         ad = astrodata.open(_file)
#         assert ad.phu['INSTRUME'] == 'F2'
#         assert ad.instrument() == ad.phu['INSTRUME']
#
#
# def test_can_return_ad_length(f2_files):
#     for _file in f2_files:
#         ad = astrodata.open(_file)
#         assert len(ad) == 1
#
#
# def test_slice_range(f2_files):
#     for _file in f2_files:
#         ad = astrodata.open(_file)
#         metadata = ('SCI', 2), ('SCI', 3)
#         slc = ad[1:]
#
#         assert len(slc) == 0
#
#     for ext, md in zip(slc, metadata):
#         assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md
#
#
# def test_read_a_keyword_from_phu(f2_files):
#     for _file in f2_files:
#         ad = astrodata.open(_file)
#         assert ad.phu['INSTRUME'].strip() == 'F2'
#
#
# def test_read_a_keyword_from_hdr(f2_files):
#     for _file in f2_files:
#         ad = astrodata.open(_file)
#         try:
#             assert ad.hdr['CCDNAME'] == 'F2'
#         except KeyError:
#             # KeyError only accepted if it's because headers out of range
#             assert len(ad) == 1


if __name__ == "__main__":
    pytest.main()
