{   'addDQ':{'postpend':
                   {
                    'default': '_dq' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':True 
                    },
                    'fl_saturated':
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     },
                     'fl_nonlinear':
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     }
                    },   
    'addVAR':{'postpend':
                   {
                    'default': '_var' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
                    'fl_saturated':
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     },
                     'fl_nonlinear':
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     }
                    },
    'ADUtoElectrons':{'postpend':
                      {
                    'default': '_aduToElect' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':True 
                       }
                      },

    'combine':{'postpend':
                   {
                    'default': '_comb' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
                      'fl_dqprop':
                   {
                    'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True            
                    },
                    'fl_vardq':
                   {
                   'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True 
                    },   
                    'method':
                   {
                   'default': 'average' , 
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False  
                    },   
                    },
    'measureIQ':{'function':
                 {
                  'default': 'both' , # can be moffat/gauss/both
                  'recipeOverride': True,
                  'type': 'str',
                  'userOverride':False
                  },
                  'display':
                   {
                   'default': True , 
                   'recipeOverride': True,
                   'type': 'bool',
                   'userOverride':True 
                    },   
                    'mosaic':
                   {
                   'default': True , 
                   'recipeOverride': True,
                   'type': 'bool',
                   'userOverride':True 
                    }, 
                    'qa':
                   {
                   'default': True , 
                   'recipeOverride': True,
                   'type': 'bool',
                   'userOverride':True 
                    },     
                 },            
    'pause': {'message':
              { 'default':'Pausing Reduction by Control System Request',
               'type':'string',
               'a':'default comes first, the rest alphabetically',
               'note1':'these are just test parameters...',
               'note2':"pause doesn't need a 'message' parameter"
               }
              },      
    'showParameters': {'test': 
                   {
                    'default': True,
                    'recipeOverride': False,
                    'uiLevel': 'debug',
                    'userOverride':True,
                    'type': 'bool',
                    'tags': ['test', 'iraf']
                    },
                 
                 'otherTest':
                    {'default': False,
                     'userOverride':True,

                    },
                 
                 'otherTest2':
                    {
                     'userOverride':True,
                     'tags':['test', 'wcslib']
                    }
                },
    'standardizeHeaders':{'postpend':
                   {
                    'default': '_headers' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    }     
                          },
    'standardizeStructure':{'postpend':
                   {
                    'default': '_struct' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
                    'addMDF':
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     }     
                          },                      
    'validateData':{'postpend':
                   {
                    'default': '_validated' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
                    'repair':
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     }     
                          },
    'writeOutputs':{'strip':
                    {
                     'default': False , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True 
                     },
                     'clobber':
                    {
                     'default': False , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True 
                     },
                     'postpend':
                     {
                     'default': '' , 
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':True 
                     },
                     'prepend':
                     {
                     'default': '' , 
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':True 
                     },
                    'outfilename':
                     {
                     'default': '' , 
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':True 
                     },
                    },
}    

