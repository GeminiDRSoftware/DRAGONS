{   "showParams": {"test": 
                   {
                    "default": "GMOS_SPECT_SETTING",
                    "recipeOverride": True,
                    "uiLevel": "debug",
                    "userOverride":True
                    
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
    "makeFringeFrame": {
                'outpref':
                   {
                    'default': '_fringe' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
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
                   'default': 'median' , 
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False  
                     },  
                    },   
    "fringeCorrect": {
                'outpref':
                   {
                    'default': '_fringecorrected' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
            'statsec':
                   {
                    'default': '' , 
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
                      
                'fl_statscale':
                   {
                   'default': True , 
                      'recipeOverride': True,
                      'type': 'bool',
                      'userOverride':True 
                    },
                    
                'scale':
                   {
                   'default': 0.0 , 
                      'recipeOverride': True,
                      'type': 'float',
                      'userOverride':False  
                     },  
                    },             
}
