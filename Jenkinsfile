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
    TEST_PATH = "$WORKSPACE/test_path/"
  }

  stages {
    stage ("Code pull"){
      steps{
        checkout scm
      }
    }
    stage ("Download and Install Anaconda") {
      steps {
        sh 'python .jenkins/download_test_data.py'
      }
    } // stage: download and install anaconda

    stage ("Build and Test Environment") {
      steps {
        sh '''conda env create --quiet --file .jenkins/conda_venv.yml -n ${BUILD_TAG}
              source activate ${BUILD_TAG}
              pip list
              which pip
              which python
              python -c "import future"
        '''
      }
    } // stage: build environment

    stage('Download test data') {
      steps {
        sh '''  source activate ${BUILD_TAG}
                pip install pycurl
                python .jenkins/download_test_data.py
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
                pylint --disable=C astrodata || true
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
                pytest astrodata --ad_test_data_path ${TEST_PATH} \\
                  --junit-xml test-reports/results.xml
                pytest recipe_system gemini_instruments \\
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
