{   "showParams": {"test": 
                   {
                    "default": True,
                    "recipeOverride": False,
                    "uiLevel": "debug",
                    "userOverride":True,
                    "type": "bool"
                    
                    },
                 
                 "otherTest":
                    {"default": "default",
                     "userOverride":True}
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
                    }
}    

