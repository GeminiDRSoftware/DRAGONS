#
#                                                                  gemini_python
#
#                                                        recipe_system.reduction
#                                                               reduceActions.py
# ------------------------------------------------------------------------------
"""
This module provides a number "action" classes, subclassed from the
argparse.Action class. These classes only override the __call__() method. This
actions class library supplies ad hoc functionality to DPDG requirements on the
reduce command line interface.

Action classes provided:

    PosArgAction          - positional argument
    BooleanAction         - optional switches
    UnitaryArgumentAction - single value options
    ParameterAction       - user parameters (-p, --param)
    CalibrationAction     - user calibration services (--user_cal)

Becuase of requirements on the reduce interface, any new reduce options should
specify one of these actions in the add_argument() call. But only one (1)
PosArgAction should occur in a given parser.

These actions may be used in the add_argument() method call, such as,

    parser.add_argument('-f', '--foo', action=BooleanAction,
                         help="Switch on foo.")

"""
from argparse import Action


class PosArgAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values:
            setattr(namespace, self.dest, values)
        return


class BooleanAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # 'values' is a list, which may have accumulated pos args
        _pos_args = []
        _switch_state = bool(getattr(namespace, self.dest))
        _pos_args.extend([f for f in values if ".fits" in f])

        # Configure namespace w new files
        if _pos_args:
            setattr(namespace, 'files', _pos_args)

        # Toggle switch.
        setattr(namespace, self.dest, not _switch_state)
        return


class UnitaryArgumentAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # 'values' is a list, which may have accumulated pos args
        _pos_args = []
        _par_args = []
        _extant_pos_args = getattr(namespace, 'files')
        _extant_par_args = getattr(namespace, self.dest)

        for value in values:
            if ".fits" in value:
                _pos_args.extend([value])
            else:
                _par_args.extend([value])

        # set new pos args
        if _pos_args:
            setattr(namespace, 'files', _pos_args)

        # Received (new) unitary argument types
        # override any previous namespace self.dest
        setattr(namespace, self.dest, _par_args)
        return


class UploadArgumentAction(UnitaryArgumentAction):
    def __call__(self, parser, namespace, values, option_string=None):
        valid = ["metrics", "calibs", "science"]

        # 'values' is a list, which may have accumulated pos args
        _pos_args = []
        _par_args = []
        _extant_pos_args = getattr(namespace, 'files')
        _extant_par_args = getattr(namespace, self.dest)

        for value in values:
            if value in valid:
                _par_args.extend([value])
            else:
                _pos_args.extend([value])

        # set new pos args
        if _pos_args:
            setattr(namespace, 'files', _pos_args)

        # Received (new) unitary argument types
        # override any previous namespace self.dest
        setattr(namespace, self.dest, _par_args)
        return


class ParameterAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # 'values' is a list, which may have accumulated pos args
        _pos_args = []
        _par_args = []
        _extant_pos_args = getattr(namespace, 'files')
        _extant_par_args = getattr(namespace, self.dest)

        for value in values:
            if "=" not in value:
                _pos_args.extend([value])
            else:
                _par_args.extend([value])

        # set new pos args
        if _pos_args:
            setattr(namespace, 'files', _pos_args)

        # Handle parameter args already in namespace.
        # Override only those specific parameters.
        if _par_args and not _extant_par_args:
            setattr(namespace, self.dest, _par_args)

        if _extant_par_args:
            reemed = [_extant_par_args.remove(z) for z in
                      [x for x in _extant_par_args if x.split('=')[0] in
                       [y.split('=')[0] for y in _par_args]]
            ]

            print("Overriding", len(reemed), "parameter(s).\n")
            _extant_par_args.extend(_par_args)
            setattr(namespace, self.dest, _extant_par_args)
        return


class CalibrationAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # 'values' is a list, which may have accumulated pos args
        _pos_args = []
        _cal_args = []
        _extant_pos_args = getattr(namespace, 'files')
        _extant_cal_args = getattr(namespace, self.dest)

        for value in values:
            if ":" not in value:
                _pos_args.extend([value])
            else:
                _cal_args.extend([value])

        # set new pos args
        if _pos_args:
            setattr(namespace, 'files', _pos_args)

        # Handle cal args already in namespace.
        # Override specific parameters.

        if _cal_args and not _extant_cal_args:
            setattr(namespace, self.dest, _cal_args)

        if _extant_cal_args:
            reemed = [_extant_cal_args.remove(z) for z in
                      [x for x in _extant_cal_args if x.split(':')[0] in
                       [y.split(':')[0] for y in _cal_args]]
            ]

            print("Overriding", len(reemed), "calibration source(s).\n")
            _extant_cal_args.extend(_cal_args)
            setattr(namespace, self.dest, _extant_cal_args)
        return
