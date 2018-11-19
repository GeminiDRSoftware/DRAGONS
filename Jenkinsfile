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
    stage ("Check Conda") {
      steps {
        echo "Verifying docker environment"
        sh 'ls $HOME'
        sh 'echo $CONDA_HOME'
        sh 'echo $0'
      } // steps
    } // stage Check Conda
  } // stages
} // pipeline
