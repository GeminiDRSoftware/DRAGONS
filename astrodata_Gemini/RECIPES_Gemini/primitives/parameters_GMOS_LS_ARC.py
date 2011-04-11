{'wavecal':{
    # debug characters
    'debug':{
        'default'       : '',
        'recipeOverride': True,
        'type'          : 'str',
        'userOverride'  : True,
        },
    # Reference file with reference lines
    'reffile':{
        'default'       : 'cuar.dat' ,
        'recipeOverride': True,
        'type'          : 'str',
        'userOverride'  : True,
        },
    # Write database file with wavelength solution
    'wrdb':{
        'default'       : False, 
        'recipeOverride': True,
        'type'          : 'bool',
        'userOverride'  : False,
        },
    # Fitting function to use
    'fitfunction':{
        'default'       : 'chebyshev',
        'recipeOverride': True,
        'type'          : 'str',
        'userOverride'  : True,
        },
    # Fitting function order
    'fitorder':{
        'default'       : 4,
        'recipeOverride': True,
        'type'          : 'int',
        'userOverride'  : True,
        },
    # Number of target features
    'ntmax':{
        'default'       : 50,
        'recipeOverride': True,
        'type'          : 'int',
        'userOverride'  : True,
        },
    # Full-width at the base 
    'fwidth':{
        'default'       : 10,
        'recipeOverride': True,
        'type'          : 'int',
        'userOverride'  : True,
        },
    # Coordinate list matching limit
    'match':{
        'default'       : -6,
        'recipeOverride': True,
        'type'          : 'int',
        'userOverride'  : True,
        },
    # The maximum distance, in pixels, allowed when 
    'cradius':{
        # defining a new line.
        'default'       : 12,
        'recipeOverride': True,
        'type'          : 'int',
        'userOverride'  : True,
        },
    # clip*stdeviacion rejection limit for refiting
    'clip':{
        'default'       : 5,
        'recipeOverride': True,
        'type'          : 'int',
        'userOverride'  : True,
        },
    # clip*stdeviacion rejection limit for refiting
    'nsum':{
        'default'       : 10,
        'recipeOverride': True,
        'type'          : 'int',
        'userOverride'  : True,
        },
    # peaks: Minimum pixel separation
    'minsep':{
        'default'       : 2,
        'recipeOverride': True,
        'type'          : 'int',
        'userOverride'  : True,
        },
    # logfile
    'logfile':{
        'default'       : '',
        'recipeOverride': True,
        'type'          : 'str',
        'userOverride'  : True,
        },
    },
 }
