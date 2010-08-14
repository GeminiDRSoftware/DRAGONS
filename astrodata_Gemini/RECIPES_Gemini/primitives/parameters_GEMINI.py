{   "showParams": {"test": 
                   {
                    "default": True,
                    "recipeOverride": False,
                    "uiLevel": "debug",
                    "userOverride":True,
                    "type": "bool",
                    "tags": ["test", "iraf"]
                    },
                 
                 "otherTest":
                    {"default": False,
                     "userOverride":True,
                     
                    },
                 
                 "otherTest2":
                    {
                     "userOverride":True,
                     "tags":["test", "wcslib"]
                    }
                },
    "pause": {"message":
              { "default":"Pausing Reduction by Control System Request",
               "type":"string",
               "a":"default comes first, the rest alphabetically",
               "note1":"these are just test parameters...",
               "note2":"pause doesn't need a 'message' parameter"
               }
              },
    "validateData":{'outsuffix':
                   {
                    'default': '_validated' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
                    "repair":
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     }     
                          },
    "standardizeStructure":{'outsuffix':
                   {
                    'default': '_struct' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
                    "addMDF":
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     }     
                          },
    "standardizeHeaders":{'outsuffix':
                   {
                    'default': '_prepared' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    }     
                          },
    "calculateVAR":{'outsuffix':
                   {
                    'default': '_vardq' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
                    "fl_saturated":
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     },
                     "fl_nonlinear":
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     }
                    },
    "calculateDQ":{'outsuffix':
                   {
                    'default': '_vardq' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
                    "fl_saturated":
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     },
                     "fl_nonlinear":
                    {
                      'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True                                      
                     }
                    },
    "combine":{'outpref':
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
                          },
    'ADUtoElectrons':{'outpref':
                      {
                    'default': '_aduToElect' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':True 
                       }
                      }
}    

