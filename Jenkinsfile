pipeline {
  agent any
  stages {
    stage('Installation') {
      steps {
        echo 'Define important variables'
        sh '''echo $WORKSPACE
PACKAGE_NAME="dragons"
CONDA_HOME=$WORKSPACE/miniconda
PYENV_HOME=$WORKSPACE/.pyenv'''
        echo 'Check if existing conda environment'
        sh '''if [ ! -d $CONDA_HOME ]; then
    /usr/local/bin/wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
	bash miniconda.sh -b -p $CONDA_HOME
fi'''
        echo 'Add conda to path'
        sh '''export PATH="$CONDA_HOME/bin:$PATH"
hash -r'''
      }
    }
  }
}