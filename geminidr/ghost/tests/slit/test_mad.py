from geminidr.ghost.primitives_ghost_slit import _mad


def test__mad_fullarray():
    """
    Checks to make:

    - Pass in some known data, check the MAD is computed correctly
    """
    # Create a simple array where the MAD is easily known
    test_array = [1., 1., 3., 5., 5.]
    test_array_mad = 2.
    assert abs(_mad(test_array) -
               test_array_mad) < 1e-5, 'MAD computation failed ' \
                                       '(expected: {}, ' \
                                       'computed: {})'.format(
        test_array_mad, _mad(test_array),
    )


def test__mad_cols():
    """
    Checks to make:

    - Check across axes as well
    """
    # Create a simple test array
    test_array = [
        [1., 2., 3., ],
        [4., 6., 8., ],
        [5., 10., 15., ],
    ]

    test_array_mad_cols = [1., 4., 5., ]
    assert sum([abs(_mad(test_array, axis=0)[i] -
                    test_array_mad_cols[i]) < 1e-5
                for i in
                range(len(test_array_mad_cols))]) == \
           len(test_array_mad_cols), 'MAD computation failed ' \
                                     '(axis 0) ' \
                                     '(expected: {}, ' \
                                     'computed: {})'.format(
        test_array_mad_cols, _mad(test_array, axis=0),
    )


def test__mad_rows():
    """
    Checks to make:

    - Check across axes as well
    """
    # Create a simple test array
    test_array = [
        [1., 2., 3., ],
        [4., 6., 8., ],
        [5., 10., 15., ],
    ]

    test_array_mad_rows = [1., 2., 5., ]
    assert sum([abs(_mad(test_array, axis=1)[i] -
                    test_array_mad_rows[i]) < 1e-5
                for i in
                range(len(test_array_mad_rows))]
               ) == len(test_array_mad_rows), 'MAD computation failed ' \
                                              '(axis 1) ' \
                                              '(expected: {}, ' \
                                              'computed: {})'.format(
        test_array_mad_rows, _mad(test_array, axis=1),
    )
