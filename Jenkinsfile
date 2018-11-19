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
              echo $HOME
              echo $CONDA_HOME
              env
              '''
        sh '''if [ -f /.dockerenv ]; then
                echo "I'm inside matrix ;(";
              else
                echo "I'm living in real world!";
              fi
              '''
        sh '''if [ ! -d $CONDA_HOME ]; then
                curl --silent https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh --output anaconda.sh
                /bin/bash -c "./anaconda.sh -b -p ~/anaconda/"
              fi
              '''
        sh '''ls $CONDA_HOME/bin
              export PATH="$CONDA_HOME/bin:$PATH"
              hash -r
              
              conda config --set always_yes yes --set changeps1 no
              conda config --add channels http://ssb.stsci.edu/astroconda
              conda update -q conda
              '''
      } // steps
    } // stage Check Conda
    stage ("Build Environment") {
      steps {
        sh '''
              export PATH="$CONDA_HOME/bin:$PATH"
              conda create --yes -n ${BUILD_TAG} python
              source activate ${BUILD_TAG}
              conda install coverage pytest
              conda install -c omnia behave
              conda install -c conda-forge twine
              conda install -c chroxvi radon
        '''
      }
    }
    stage('Test environment') {
      steps {
        sh '''
              export PATH="$CONDA_HOME/bin:$PATH"
              source activate ${BUILD_TAG}
              pip list
              which pip
              which python
              '''
      }
    }
    stage('Static code metrics') {
      steps {
        echo "Raw metrics"
        sh  '''
                export PATH="$CONDA_HOME/bin:$PATH"
                source activate ${BUILD_TAG}
                radon raw --json irisvmpy/ > raw_report.json
                radon cc --json irisvmpy/ > cc_report.json
                radon mi --json irisvmpy/ > mi_report.json
                '''
        echo "Code Coverage"
        sh  '''
                export PATH="$CONDA_HOME/bin:$PATH"
                source activate ${BUILD_TAG}
                coverage run --source=astrodata,gemini_instruments,gempy,recipe_system
                python -m coverage xml -o ./reports/coverage.xml
                '''
        echo "PEP8 style check"
        sh  '''
                export PATH="$CONDA_HOME/bin:$PATH"
                source activate ${BUILD_TAG}
                pylint --disable=C irisvmpy || true
                '''
      }
      post{
        always{
          step([$class: 'CoberturaPublisher',
              autoUpdateHealth: false,
              autoUpdateStability: false,
              coberturaReportFile: 'reports/coverage.xml',
              failNoReports: false,
              failUnhealthy: false,
              failUnstable: false,
              maxNumberOfBuilds: 10,
              onlyStable: false,
              sourceEncoding: 'ASCII',
              zoomCoverageChart: false])
        }
      }
    }
    stage('Unit tests') {
      steps {
        sh  ''' export PATH="$CONDA_HOME/bin:$PATH"
                source activate ${BUILD_TAG}
                python -m pytest --verbose --junit-xml test-reports/results.xml
                '''
      }
      post {
        always {
          // Archive unit tests for the future
          junit (
            allowEmptyResults: true,
            testResults: 'test-reports/results.xml'
            //, fingerprint: true
            )
        }
      }
    }
    stage('Build package') {
      when {
        expression {
          currentBuild.result == null || currentBuild.result == 'SUCCESS'
        }
      }
      steps {
        sh  '''
                export PATH="$CONDA_HOME/bin:$PATH"
                source activate ${BUILD_TAG}
                python setup.py sdist bdist_wheel
            '''
      }
      post {
        always {
        // Archive unit tests for the future
        archiveArtifacts (allowEmptyArchive: true,
                            artifacts: 'dist/*whl',
                            fingerprint: true)
        }
      }
    }
  }
  post {
    always {
      sh '''
        export PATH="$CONDA_HOME/bin:$PATH"
        conda remove --yes -n ${BUILD_TAG} --all
        '''
    }
    failure {
      echo "Send e-mail, when failed"
    }
  }
}
