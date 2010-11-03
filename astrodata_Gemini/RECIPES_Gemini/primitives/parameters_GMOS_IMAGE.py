{  'fringeCorrect': {
                'postpend':
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
    'makeFringeFrame': {
                'postpend':
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
}
