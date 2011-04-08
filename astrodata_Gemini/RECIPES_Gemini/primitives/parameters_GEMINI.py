{   'addDQ':{'suffix':
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
    'addVAR':{'suffix':
                   {
                    'default': '_var' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    }
                    },
    'aduToElectrons':{'suffix':
                      {
                    'default': '_aduToElect' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':True 
                       }
                      },

    'combine':{'suffix':
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
                    
     'flatCorrect':{'suffix':
                   {
                    'default': '_comb' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },               
                    },
       
    'getCal': {
                'source':
                {
                    'default':'all',
                    'type':'str'
                },
                'caltype':
                {
                    'default': None,
                    'type':'str'
                }
              },
    'getList':{'purpose':
                   {
                    'default': '' ,
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
                    'qa':
                   {
                   'default': True , # A flag to use a grid of sub-windows for detecting the sources in the image frames, rather than the entire frame all at once.
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
              
    'addToList':{'purpose':
                   {
                    'default': '' ,
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },               
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
                
    'showList':{'purpose':
                   {
                    'default': '' ,
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },               
                    },
    
                          
    'standardizeStructure':{'suffix':
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
                          
      'storeProcessedBias':{'clob':
                          {
                            'default': False,
                            'recipeOverride': True,
                            'type': 'bool',
                            'userOverride':True  
                           }
                          } ,
    'storeProcessedFlat':{'clob':
                          {
                            'default': False,
                            'recipeOverride': True,
                            'type': 'bool',
                            'userOverride':True  
                           }
                          } ,
    'subtractDark':{'suffix':
                   {
                    'default': '_subdark' , #value to be post pended to this primitives outputs
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':False 
                    },
                    },
                                            
    'validateData':{'suffix':
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
                     'suffix':
                     {
                     'default': '' , 
                      'recipeOverride': True,
                      'type': 'str',
                      'userOverride':True 
                     },
                     'prefix':
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

