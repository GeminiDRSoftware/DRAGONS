{
 'wavecal':{'debug': {# debug characters
                       'default': '' , 
                       'recipeOverride': True,
                       'type': 'str',
                       'userOverride':True
                      },
            'reffile': {# Reference file with reference lines
                       'default': 'cuar.dat' ,
                       'recipeOverride': True,
                       'type': 'str',
                       'userOverride':True
                      },
              'wrdb': {# Write database file with wavelength solution
                       'default': False, 
                       'recipeOverride': True,
                       'type': 'bool',
                       'userOverride':False
                      },
       'fitfunction': {# Fitting function to use
                       'default': 'chebyshev',
                       'recipeOverride': True,
                       'type': 'str',
                       'userOverride':True
                      },
          'fitorder': {# Fitting function order
                       'default': 4,
                       'recipeOverride': True,
                       'type': 'int',
                       'userOverride':True
                      },
             'ntmax': {# Number of target features
                       'default': 50,
                       'recipeOverride': True,
                       'type': 'int',
                       'userOverride':True
                      },
             'fwidth': {# Full-width at the base 
                       'default': 10,
                       'recipeOverride': True,
                       'type': 'int',
                       'userOverride':True
                      },
             'match': {# Coordinate list matching limit
                       'default': -6,
                       'recipeOverride': True,
                       'type': 'int',
                       'userOverride':True
                      },
             'cradius': {# The maximum distance, in pixels, allowed when 
                         # defining a new line.
                       'default': 12,
                       'recipeOverride': True,
                       'type': 'int',
                       'userOverride':True
                      },
             'clip': {# clip*stdeviacion rejection limit for refiting
                       'default': 5,
                       'recipeOverride': True,
                       'type': 'int',
                       'userOverride':True
                      },
             'nsum': {# clip*stdeviacion rejection limit for refiting
                       'default': 10,
                       'recipeOverride': True,
                       'type': 'int',
                       'userOverride':True
                      },
            'minsep': {# peaks: Minimum pixel separation
                       'default': 2,
                       'recipeOverride': True,
                       'type': 'int',
                       'userOverride':True
                      },
            'logfile': {# logfile
                       'default': '',
                       'recipeOverride': True,
                       'type': 'str',
                       'userOverride':True
                      },
                },

}
