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
    PATH = "$JENKINS_HOME/anaconda3/bin:$PATH"
  }

  stages {
    stage ("Code pull"){
      steps{
        checkout scm
      }
    }
    stage ("Download and Install Anaconda") {
      steps {
        sh '''if ! [ "$(command -v conda)" ]; then
                echo "Conda is not installed - Downloading and installing"

                curl https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh \\
                  --output anaconda.sh --silent

                chmod a+x anaconda.sh
                ./anaconda.sh -u -b -p $JENKINS_HOME/anaconda3/

                conda config --add channels http://ssb.stsci.edu/astroconda
                conda update --quiet conda
              fi
              '''
      }
    } // stage: download and install anaconda

    stage ("Build Environment") {
      steps {
        sh '''conda create --yes -n ${BUILD_TAG} python
              source activate ${BUILD_TAG}
              conda install coverage pytest
              conda install -c omnia behave
              conda install -c conda-forge twine
              conda install -c chroxvi radon
              conda install future cython
        '''
      }
    } // stage: build environment

    stage('Test environment') {
      steps {
        sh '''source activate ${BUILD_TAG}
              pip list
              which pip
              which python
              '''
      }
    } // stage: test environment

    stage('Static code metrics') {
      steps {
        echo "Code Coverage"
        sh  ''' source activate ${BUILD_TAG}
                coverage run setup.py build
                python -m coverage xml -o ./reports/coverage.xml
                '''
        echo "PEP8 style check"
        sh  ''' source activate ${BUILD_TAG}
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
    } // stage: static code metrics

    stage('Unit tests') {
      steps {
        sh  ''' source activate ${BUILD_TAG}
                pytest astrodata recipe_system gemini_instruments \\
                  --junit-xml test-reports/results.xml
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
    } // stage: unit tests

    stage('Build package') {
      when {
        expression {
          currentBuild.result == null || currentBuild.result == 'SUCCESS'
        }
      }
      steps {
        sh  ''' source activate ${BUILD_TAG}
                python setup.py sdist bdist_egg
            '''
      }
      post {
        always {
        // Archive unit tests for the future
        archiveArtifacts (allowEmptyArchive: true,
                            artifacts: 'dist/*whl',
                            fingerprint: true)
        }
      } // post
    } // stage: build package
  } // stages

  post {
    always {
      sh 'conda remove --yes -n ${BUILD_TAG} --all'
    }
    failure {
      echo "Send e-mail, when failed"
    }
  } // post

} // pipeline
