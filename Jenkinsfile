#!/usr/bin/env groovy
pipeline {

  agent any

  triggers {
    pollSCM('*/5 * * * 1-5')
  }

  options {
    skipDefaultCheckout(true)

    // Keep the 10 most recent builds
    buildDiscarder(logRotator(numToKeepStr: '10'))

    timestamps()
  }

  environment {
    CONDA_HOME="$HOME/anaconda"
  }

  stages {
    stage ("Code pull"){
      steps{
        checkout scm
      }
    }
    stage ("Anaconda") {
      steps {
        sh '''
          /bin/bash -c
          if [ ! -d $CONDA_HOME ]; then
            curl --silent https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh --output anaconda.sh
            /bin/bash -c "anaconda.sh -b -p $HOME/anaconda/"
          fi
          '''
      } // steps
    } // stage Check Conda
  } // stages
} // pipeline
