pipeline {
  agent any
    stages {
      stage('Installation') {
        steps {
          sh '''
            echo $WORKSPACE
            PACKAGE_NAME="dragons"
            CONDA_HOME=$WORKSPACE/miniconda
            PYENV_HOME=$WORKSPACE/.pyenv
	    
            echo $CONDA_HOME
            if [ ! -d $CONDA_HOME ]; then
              /usr/local/bin/wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
              bash miniconda.sh -b -p $CONDA_HOME
              fi
        
            export PATH="$CONDA_HOME/bin:$PATH"
            echo $PATH
            hash -r

            conda config --set always_yes yes --set changeps1 no
            conda config --add channels http://ssb.stsci.edu/astroconda
            conda update -q conda
	
            echo $PYENV_HOME
            if [ ! -d $PYENV_HOME ]; then
              conda create --quiet --yes --prefix $PYENV_HOME numpy scipy matplotlib pandas astropy ccdproc cython
              fi
 
            source activate $PYENV_HOME
	     
            conda install cython
            pip install --quiet --upgrade pip
            pip install --quiet -r requirements.txt

            cd $WORKSPACE/gempy/library
            cythonize -a -i cyclip.pyx
            
            cd $WORKSPACE
            pip install --quiet .
      }
    }
  }
}
