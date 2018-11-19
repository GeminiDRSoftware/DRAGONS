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
        sh '''echo "Verifying conda installation ---"
              cat /etc/os-release
              echo $HOME
              ls $HOME
              echo $CONDA_HOME
              '''
        sh '''if [ -f /.dockerenv ]; then
                echo "I'm inside matrix ;(";
              else
                echo "I'm living in real world!";
              fi
              '''
      } // steps
    } // stage Check Conda
  } // stages
} // pipeline
