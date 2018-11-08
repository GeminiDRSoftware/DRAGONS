pipeline {
  agent any
  stages {
    stage('Installation') {
      steps {
        
	echo 'Define important variables'
        sh '''
            echo $WORKSPACE
            PACKAGE_NAME="dragons"
            CONDA_HOME=$WORKSPACE/miniconda
            PYENV_HOME=$WORKSPACE/.pyenv
	'''
        
	echo 'Check if existing conda environment'
        sh '''
	    if [ ! -d $CONDA_HOME ]; then
	        /usr/local/bin/wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
                bash miniconda.sh -b -p $CONDA_HOME
            fi
	'''
        
	echo 'Add conda to path'
        sh '''
	    export PATH="$CONDA_HOME/bin:$PATH"
	    hash -r
	   '''
	
	echo 'Setting conda to use astroconda'      
	sh '''      
            conda config --set always_yes yes --set changeps1 no
            conda config --add channels http://ssb.stsci.edu/astroconda
            conda update -q conda
	    '''
	
	echo 'Check if there is an existing virtual environment'
	sh '''
	    if [ ! -d $PYENV_HOME ]; then
                conda create --quiet --yes --prefix $PYENV_HOME numpy scipy matplotlib pandas astropy ccdproc cython
            fi
	    '''
	
	echo 'Activate virtual environment'
        sh 'source activate $PYENV_HOME'
	      
	echo 'Install extra packages'
	sh '''
            conda install cython
            pip install --quiet --upgrade pip
            pip install --quiet -r requirements.txt
        ''' 
	
        echo 'Compile libraries that use Cython'
	sh '''
	    cd $WORKSPACE/gempy/library
            cythonize -a -i cyclip.pyx
        '''
	      
	echo 'Install DRAGONS'
	sh '''
            cd $WORKSPACE
	    pip install --quiet .  # where your setup.py lives
	'''
      }
    }
  }
}
