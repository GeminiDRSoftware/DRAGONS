{'addBPM':{
    'suffix':{
        # String to be post pended to the output of addBPM
        'default'       : '_bpmAdded',
        'recipeOverride': True,
        'type'          : 'str',
        'userOverride'  : False,
        },
    },
'standardizeInstrumentHeaders':{
    'suffix':{
        # String to be post pended to the output of
        # standardizeInstrumentHeaders
        'default'       : '_sdzInstHdrs',
        'recipeOverride': True,
        'type'          : 'str',
        'userOverride'  : False,
        },
    },
 'standardizeInstrumentStructure':{
    'suffix':{
        # String to be post pended to the output of
        # standardizeInstrumentStructure
        'default'       : '_sdzInstStruct',
        'recipeOverride': True,
        'type'          : 'str',
        'userOverride'  : False,
        },
    'addMDF':{
        'default'       : True,
        'recipeOverride': True,
        'type'          : 'bool',
        'userOverride'  : True,
        },
    },
'validateInstrumentData':{
    'suffix':{
        # String to be post pended to the output of validateInstrumentData
        'default'       : '_validatedInst',
        'recipeOverride': True,
        'type'          : 'str',
        'userOverride'  : False,
        },
    'repair':{
        'default'       : True,
        'recipeOverride': True,
        'type'          : 'bool',
        'userOverride'  : True,
        },
    },
}
