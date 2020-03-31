#!/usr/bin/env groovy
/*
 * Jenkins Pipeline for DRAGONS
 *
 * by Bruno C. Quint
 *
 * Required Plug-ins:
 * - CloudBees File Leak Detector
 * - Cobertura Plug-in
 * - Warnings NG
 */

@Library('dragons_ci@master') _

pipeline {

    agent {
        label 'master'
    }

    triggers {
        pollSCM('H/6 * * * *')  // Polls Source Code Manager every six hours
    }

    options {
        skipDefaultCheckout(true)
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }

    environment {
        PATH = "$JENKINS_HOME/anaconda3/bin:$PATH"
        MPLBACKEND = "agg"
        // CONDA_ENV_FILE = ".jenkins/conda_py3env_stable.yml"
        // CONDA_ENV_NAME = "py3_stable"
        // PYTEST_ARGS = "--remote-data=any --basetemp=/data/jenkins/dragons/outputs"
    }

    stages {

        stage ("Prepare"){

            steps{
                sendNotifications 'STARTED'
                checkout scm
                sh 'git clean -fxd'
                sh 'mkdir plots reports'
                sh '.jenkins/scripts/download_and_install_anaconda.sh'
            }

        }

        stage('Code Metrics') {

            when {
                branch 'master'
            }
            steps {
                sh 'tox -e check'
            }
            post {
                success {
                    recordIssues(
                        enabledForFailure: true,
                        tools: [
                            pyLint(pattern: '**/reports/pylint.log'),
                            pyDocStyle(pattern: '**/reports/pydocstyle.log')
                        ]
                    )
                }
            }

        }

        stage('Unit tests Python 3.6') {
          steps {
            echo "Running tests with Python 3.6 and older dependencies"
            sh 'tox -e py36-unit-olddeps -v -- --junit-xml reports/unittests_results.xml'
            echo "Reportint coverage to CodeCov"
            sh 'tox -e codecov -- -F unit'
          }
        }

        stage('Unit tests Python 3.7') {
          steps {
            echo "Running tests with Python 3.7"
            sh 'tox -e py37-unit -v -- --junit-xml reports/unittests_results.xml'
            echo "Reportint coverage to CodeCov"
            sh 'tox -e codecov -- -F unit'
          }
        }

        stage('Integration tests') {
          // when {
          //     branch 'master'
          // }
          steps {
            echo "Integration tests"
            sh 'tox -e py36-integ -v -- --junit-xml reports/integration_results.xml'
            echo "Reporting coverage"
            sh 'tox -e codecov -- -F integration'
          }
        }

        stage('GMOS LS Tests') {
          steps {
            echo "Running tests"
            sh 'tox -e py36-gmosls -v -- --junit-xml reports/unittests_results.xml'
            echo "Reporting coverage"
            sh 'tox -e codecov -- -F gmosls'
          }  // end steps
          post {
            always {
              echo "Running 'archivePlots' from inside GmosArcTests"
              archiveArtifacts artifacts: "plots/*", allowEmptyArchive: true
            }  // end always
          }  // end post
        }  // end stage

    }
    post {
        always {
          junit (
            allowEmptyResults: true,
            testResults: 'reports/*_results.xml'
            )
        }
        success {
//             sh  '.jenkins/scripts/build_sdist_file.sh'
//             sh  'pwd'
//             echo 'Make tarball available'
            sendNotifications 'SUCCESSFUL'
        }
        failure {
            sendNotifications 'FAILED'
        }
    }
}
