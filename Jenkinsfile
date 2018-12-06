#!/usr/bin/env groovy
pipeline {

  agent { label 'centos7'}

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
    PATH="~/anaconda3/bin:$PATH"
  }

  stages {
    stage ("Code pull"){
      steps{
        checkout scm
      }
    }
    stage ("Build Environment") {
      steps {
        sh '''conda create --yes -n ${BUILD_TAG} python
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
        sh '''source activate ${BUILD_TAG}
              pip list
              which pip
              which python
              '''
      }
    }
    stage('Static code metrics') {
      steps {
        echo "Raw metrics"
        sh  ''' source activate ${BUILD_TAG}
                radon raw --json irisvmpy/ > raw_report.json
                radon cc --json irisvmpy/ > cc_report.json
                radon mi --json irisvmpy/ > mi_report.json
                '''
        echo "Code Coverage"
        sh  ''' source activate ${BUILD_TAG}
                coverage run --source=astrodata,gemini_instruments,gempy,recipe_system
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
    }
    stage('Unit tests') {
      steps {
        sh  ''' source activate ${BUILD_TAG}
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
        sh  ''' source activate ${BUILD_TAG}
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
      sh 'conda remove --yes -n ${BUILD_TAG} --all'
    }
    failure {
      echo "Send e-mail, when failed"
    }
  }
}
