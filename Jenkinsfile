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
        // Polls Source Code Manager every four hours
        pollSCM('*/15 * * * *')
    }

    options {
        skipDefaultCheckout(true)
        buildDiscarder(logRotator(numToKeepStr: '5'))
        timestamps()
        timeout(time: 6, unit: 'HOURS')
    }

    environment {
        MPLBACKEND = "agg"
    }

    stages {

        stage ("Prepare"){
            steps{
                sendNotifications 'STARTED'
            }
        }

        stage('Normal tests') {
            parallel {

                stage('Unit tests') {

                    agent{
                        label "centos7"
                    }
                    environment {
                        MPLBACKEND = "agg"
                        PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
                        DRAGONS_TEST_OUT = "unit_tests_outputs/"
                        TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                        TMPDIR = "${env.WORKSPACE}/.tmp/unit/"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        sh '.jenkins/scripts/setup_agent.sh'
                        echo "Running tests with Python 3.7"
                        sh 'tox -e py37-unit -v -r -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/unittests_results.xml ${TOX_ARGS}'
                        echo "Reportint coverage to CodeCov"
                        sh 'tox -e codecov -- -F unit'
                    }
                    post {
                        always {
                            junit (
                                allowEmptyResults: true,
                                testResults: 'reports/*_results.xml'
                            )
                            echo "Delete temporary folder: ${TMPDIR}"
                            dir ( '$TMPDIR' ) { deleteDir() }
                            echo "Delete Tox Environment: .tox/py37-unit"
                            dir ( ".tox/py37-unit" ) { deleteDir() }
                        }
                        failure {
                            echo "Archiving tests results for Unit Tests"
                            sh "find ${DRAGONS_TEST_OUT} -not -name \\*.bz2 -type f -print0 | xargs -0 -n1 -P4 bzip2"
        //                             archiveArtifacts artifacts: "${DRAGONS_TEST_OUT}/**"
                        }
                        success {
                            echo "Delete Outputs folder: "
                            dir ( "${DRAGONS_TEST_OUT}" ) { deleteDir() }
                        }
                    }

                }

                stage('Integration tests') {
                    agent { label "centos7" }
                    environment {
                        MPLBACKEND = "agg"
                        PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
                        DRAGONS_TEST_OUT = "./integ_tests_outputs/"
                        TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                        TMPDIR = "${env.WORKSPACE}/.tmp/integ/"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        echo "${env.PATH}"
                        sh '.jenkins/scripts/setup_agent.sh'
                        echo "Integration tests"
                        sh 'tox -e py37-integ -v -r -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/integration_results.xml ${TOX_ARGS}'
                        echo "Reporting coverage"
                        sh 'tox -e codecov -- -F integration'
                    } // end steps
                    post {
                        always {
                            junit (
                                allowEmptyResults: true,
                                testResults: 'reports/*_results.xml'
                            )
                            echo "Delete temporary folder: ${TMPDIR}"
                            dir ( '$TMPDIR' ) { deleteDir() }
                            echo "Delete Tox Environment: .tox/py37-integ"
                            dir ( ".tox/py37-integ" ) { deleteDir() }
                        }
                        success {
                            echo "Delete Outputs folder: "
                            dir ( "${DRAGONS_TEST_OUT}" ) { deleteDir() }
                        }
                    } // end post
                } // end stage

                stage('Regression Tests') {
                    agent { label "master" }
                    environment {
                        MPLBACKEND = "agg"
                        PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
                        DRAGONS_TEST_OUT = "regression_tests_outputs"
                        TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                        TMPDIR = "${env.WORKSPACE}/.tmp/regr/"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        echo "${env.PATH}"
                        sh '.jenkins/scripts/setup_agent.sh'
                        echo "Regression tests"
                        sh 'tox -e py37-reg -v -r -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/regression_results.xml ${TOX_ARGS}'
                        echo "Reporting coverage"
                        sh 'tox -e codecov -- -F regression'
                    } // end steps
                    post {
                        always {
                            junit (
                                allowEmptyResults: true,
                                testResults: 'reports/*_results.xml'
                            )
                            echo "Delete temporary folder: ${TMPDIR}"
                            dir ( '$TMPDIR' ) { deleteDir() }
                            echo "Delete Tox Environment: .tox/py37-reg"
                            dir ( ".tox/py37-reg" ) { deleteDir() }
                        }
                        success {
                            echo "Delete Outputs folder: "
                            dir ( "${DRAGONS_TEST_OUT}" ) { deleteDir() }
                        }
                    } // end post
                }

                stage('GMOS LS Tests') {
                    agent { label "master" }
                    environment {
                        MPLBACKEND = "agg"
                        PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
                        DRAGONS_TEST_OUT = "gmosls_tests_outputs"
                        TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                        TMPDIR = "${env.WORKSPACE}/.tmp/gmosls/"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        sh '.jenkins/scripts/setup_agent.sh'
                        echo "Running tests"
                        sh 'tox -e py37-gmosls -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/gmosls_results.xml ${TOX_ARGS}'
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
                            echo "Delete temporary folder: ${TMPDIR}"
                            dir ( '$TMPDIR' ) { deleteDir() }
                            echo "Delete Tox Environment: .tox/py37-gmosls"
                            dir( '.tox/py37-gmosls' ) { deleteDir() }
                        }  // end always
                        success {
                            echo "Delete Outputs folder: "
                            dir ( "${DRAGONS_TEST_OUT}" ) { deleteDir() }
                        }
                    }  // end post
                }  // end stage

            }  // end parallel
        }

        stage('Slow Tests') {
            agent { label "master" }
            environment {
                MPLBACKEND = "agg"
                PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
                DRAGONS_TEST_OUT = "regression_tests_outputs"
                TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                TMPDIR = "${env.WORKSPACE}/.tmp/slow/"
            }
            steps {
                echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                checkout scm
                echo "${env.PATH}"
                sh '.jenkins/scripts/setup_agent.sh'
                echo "Slow tests"
                // KL: uncomment this next line to run the slow tests.
//                 sh 'tox -e py37-slow -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/slow_results.xml ${TOX_ARGS}'
                echo "Reporting coverage"
                sh 'tox -e codecov -- -F slow'
            } // end steps
            post {
                always {
                    junit (
                        allowEmptyResults: true,
                        testResults: 'reports/*_results.xml'
                    )
                    echo "Delete temporary folder: ${TMPDIR}"
                    dir ( '$TMPDIR' ) { deleteDir() }
                    echo "Delete Tox Environment: .tox/py37-slow"
                    dir( '.tox/py37-slow' ) { deleteDir() }
                }
                success {
                    echo "Delete Outputs folder: "
                    dir ( "${DRAGONS_TEST_OUT}" ) { deleteDir() }
                }
            } // end post
        }

    }
    post {
        success {
            sendNotifications 'SUCCESSFUL'
            deleteDir() /* clean up our workspace */
        }
        failure {
            sendNotifications 'FAILED'
        }
    }
}
