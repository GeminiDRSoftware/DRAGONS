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

    agent any

    triggers {
        // pollSCM('MIN HOUR DoM MONTH DoW')
        pollSCM('H H/4 * * *')  // Polls Source Code Manager every three hours
    }

    options {
        skipDefaultCheckout(true)
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
        timeout(time: 4, unit: 'HOURS')
    }

    environment {
        MPLBACKEND = "agg"
    }

    stages {

        stage ("Setup") {
            steps{

                sendNotifications 'STARTED'

                script {
                    properties([
                        parameters([
                            booleanParam(
                                defaultValue: false,
                                description: 'Create/update input files for test',
                                name: 'CREATE_INPUTS'
                            ),
                            booleanParam(
                                defaultValue: false,
                                description: 'Create/update references for test',
                                name: 'CREATE_REFS'
                            ),
                            booleanParam(
                                defaultValue: false,
                                description: 'Use branch name in inputs/refs path',
                                name: 'branch_name_in_path'
                            ),
                            string(
                                defaultValue: '',
                                name: 'TEST_MODULE',
                                trim: true
                            )
                        ])
                    ])
                }
            }
        }

//         stage('Code Metrics') {
//             when {
//                 branch 'master'
//             }
//             environment {
//                 PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
//             }
//             steps {
//                 echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
//                 checkout scm
//                 sh '.jenkins/scripts/setup_agent.sh'
//                 sh 'tox -e check'
//             }
//             post {
//                 success {
//                     recordIssues(
//                         enabledForFailure: true,
//                         tools: [
//                             pyLint(pattern: '**/reports/pylint.log'),
//                             pyDocStyle(pattern: '**/reports/pydocstyle.log')
//                         ]
//                     )
//                 }
//             }
//         }

        stage('Unit tests') {
            parallel {
            // Todo - Add jenkins user for macos machines
//                 stage('MacOS/Python 3.6') {
//                     agent{
//                         label "macos"
//                     }
//                     environment {
//                         PATH = "$CONDA_HOME/bin:$PATH"
//                     }
//                     steps {
//                         echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
//                         checkout scm
//                         sh '.jenkins/scripts/setup_agent.sh'
//                         echo "Running tests with Python 3.6 and older dependencies"
//                         sh 'tox -e py36-unit-olddeps -v -- --junit-xml reports/unittests_results.xml'
//                         echo "Reportint coverage to CodeCov"
//                         sh 'tox -e codecov -- -F unit'
//                     }
//                     post {
//                         always {
//                             junit (
//                                 allowEmptyResults: true,
//                                 testResults: 'reports/*_results.xml'
//                             )
//                         }
//                     }
//                 }

                stage('Linux/Python 3.7') {
                    agent{
                        label "centos7"
                    }
                    environment {
                        MPLBACKEND = "agg"
                        PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
                        DRAGONS_TEST_OUT = "$DRAGONS_TEST_OUT"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        sh '.jenkins/scripts/setup_agent.sh'
                        echo "Running tests with Python 3.7"
                        sh 'tox -e py37-unit -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/unittests_results.xml'
                        echo "Reportint coverage to CodeCov"
                        sh 'tox -e codecov -- -F unit'
                    }
                    post {
                        always {
                            junit (
                                allowEmptyResults: true,
                                testResults: 'reports/*_results.xml'
                            )
                        }
                    }
                }
            }
        }

        stage('Recreate Input Data') {
            when {
                expression {
                    return params.CREATE_INPUTS == true
                }
            }
            steps {
                echo "Creating/updating input data for ${params.TEST_MODULE}"
            }
        }

        stage('Recreate Reference Data') {
            when {
                expression {
                    return params.CREATE_REFS == true
                }
            }
            steps {
                echo "Creating/updating reference data for ${params.TEST_MODULE}"
            }
        }

        stage('GMOS LS Tests') {
            agent { label "master" }
            environment {
                MPLBACKEND = "agg"
                PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
                DRAGONS_TEST_OUT = "$DRAGONS_TEST_OUT"
            }
            steps {
                echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                checkout scm
                sh '.jenkins/scripts/setup_agent.sh'
                echo "Running tests"
                sh 'tox -e py36-gmosls -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/unittests_results.xml'
                echo "Reporting coverage"
                sh 'tox -e codecov -- -F gmosls'
            }  // end steps
            post {
                always {
                    echo "Running 'archivePlots' from inside GmosArcTests"
                    archiveArtifacts artifacts: "plots/*", allowEmptyArchive: true
                    junit (
                        allowEmptyResults: true,
                        testResults: 'reports/*_results.xml'
                    )
                }  // end always
            }  // end post
        }  // end stage

        stage('Integration tests') {
            agent { label "centos7" }
            environment {
                MPLBACKEND = "agg"
                PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
                DRAGONS_TEST_OUT = "$DRAGONS_TEST_OUT"
            }
            steps {
                echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                checkout scm
                echo "${env.PATH}"
                sh '.jenkins/scripts/setup_agent.sh'
                echo "Integration tests"
                sh 'tox -e py36-integ -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/integration_results.xml'
                echo "Reporting coverage"
                sh 'tox -e codecov -- -F integration'
            } // end steps
            post {
                always {
                    junit (
                        allowEmptyResults: true,
                        testResults: 'reports/*_results.xml'
                    )
                }
            } // end post
        } // end stage

    }
    post {
//         always {
//           junit (
//             allowEmptyResults: true,
//             testResults: 'reports/*_results.xml'
//             )
//         }
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
