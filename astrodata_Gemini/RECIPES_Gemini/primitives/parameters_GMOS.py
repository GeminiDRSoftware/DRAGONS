{'mosaicDetectors':{
    'tile':{
        'default'       : False,
        'recipeOverride': True,
        'type'          : 'bool',
        'userOverride'  : True,
        'tag'           : ['cl_iraf','ui_advanced'],
        },
    'interpolator':{
        'default'       : 'linear',
        'recipeOverride': True,
        'type'          : 'str',
        'userOverride'  : True,
        'tag'           : ['cl_iraf','ui_advanced'],
         },
    },
'overscanSubtract':{
    'trim':{
        'default'       : False,
        'recipeOverride': False,
        'type'          : 'bool',
        'userOverride'  : False,
        },
    'overscan_section':{
        'default'       : '[1:25,1:2304],[1:32,1:2304],[1025:1056,1:2304]',
        'recipeOverride': True,
        'type'          : 'str',
        'userOverride'  : True,
        },
    },
'standardizeStructure':{
    'add_mdf':{
        'default'       : False,
        'recipeOverride': True,
        'type'          : 'bool',
        'userOverride'  : True,
        },
    },
'validateData':{
    'repair':{
        'default'       : False,
        'recipeOverride': True,
        'type'          : 'bool',
        'userOverride'  : True,
        },
    },
 "validateData":{
    "repair":{
        "default"       : True,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
}
