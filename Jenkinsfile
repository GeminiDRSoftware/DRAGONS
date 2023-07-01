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

    //triggers {
    //   // Polls Source Code Manager every four hours
    //    pollSCM('*/15 * * * *')
    //}

    options {
        skipDefaultCheckout(true)
        buildDiscarder(logRotator(numToKeepStr: '5'))
        timestamps()
        timeout(time: 6, unit: 'HOURS')
    }

    environment {
        MPLBACKEND = "agg"
        PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
    }

    stages {

        stage ("Prepare"){
            steps{
                sendNotifications 'STARTED'
            }
        }

        stage('Pre-install') {
            agent { label "conda" }
            environment {
                TMPDIR = "${env.WORKSPACE}/.tmp/conda/"
            }
            steps {
                echo "Update the Conda base install for all on-line nodes"
                checkout scm
                sh '.jenkins/scripts/setup_agent.sh'
                echo "Create a trial Python 3.10 env, to cache new packages"
                sh 'tox -e py310-noop -v -r -- --basetemp=${DRAGONS_TEST_OUT} ${TOX_ARGS}'
            }
            post {
                always {
                    echo "Deleting conda temp workspace ${env.WORKSPACE}"
                    cleanWs()
                    dir("${env.WORKSPACE}@tmp") {
                      deleteDir()
                    }
                }
            }
        }

        stage('Quicker tests') {
            parallel {

                stage('Unit tests') {

                    agent{
                        label "centos7"
                    }
                    environment {
                        MPLBACKEND = "agg"
                        DRAGONS_TEST_OUT = "unit_tests_outputs/"
                        TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                        TMPDIR = "${env.WORKSPACE}/.tmp/unit/"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        sh '.jenkins/scripts/setup_dirs.sh'
                        echo "Running tests with Python 3.10"
                        sh 'tox -e py310-unit -v -r -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/unittests_results.xml ${TOX_ARGS}'
                        echo "Reportint coverage to CodeCov"
                        sh 'tox -e codecov -- -F unit'
                    }
                    post {
                        always {
                            junit (
                                allowEmptyResults: true,
                                testResults: '.tmp/py310-unit/reports/*_results.xml'
                            )
                            echo "Deleting Unit tests workspace ${env.WORKSPACE}"
                            cleanWs()
                            dir("${env.WORKSPACE}@tmp") {
                              deleteDir()
                            }
                        }
        //                failure {
        //                    echo "Archiving tests results for Unit Tests"
        //                    sh "find ${DRAGONS_TEST_OUT} -not -name \\*.bz2 -type f -print0 | xargs -0 -n1 -P4 bzip2"
        //                             archiveArtifacts artifacts: "${DRAGONS_TEST_OUT}/**"
        //                }
                    }
                }

                stage('Integration tests') {
                    agent { label "centos7" }
                    environment {
                        MPLBACKEND = "agg"
                        DRAGONS_TEST_OUT = "./integ_tests_outputs/"
                        TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                        TMPDIR = "${env.WORKSPACE}/.tmp/integ/"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        echo "${env.PATH}"
                        sh '.jenkins/scripts/setup_dirs.sh'
                        echo "Integration tests"
                        sh 'tox -e py310-integ -v -r -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/integration_results.xml ${TOX_ARGS}'
                        echo "Reporting coverage"
                        sh 'tox -e codecov -- -F integration'
                    } // end steps
                    post {
                        always {
                            junit (
                                allowEmptyResults: true,
                                testResults: '.tmp/py310-integ/reports/*_results.xml'
                            )
                            echo "Deleting Integration tests workspace ${env.WORKSPACE}"
                            cleanWs()
                            dir("${env.WORKSPACE}@tmp") {
                              deleteDir()
                            }
                        }
                    } // end post
                } // end stage

                stage('Regression Tests') {
                    agent { label "master" }
                    environment {
                        MPLBACKEND = "agg"
                        DRAGONS_TEST_OUT = "regression_tests_outputs"
                        TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                        TMPDIR = "${env.WORKSPACE}/.tmp/regr/"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        echo "${env.PATH}"
                        sh '.jenkins/scripts/setup_dirs.sh'
                        echo "Regression tests"
                        sh 'tox -e py310-reg -v -r -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/regression_results.xml ${TOX_ARGS}'
                        echo "Reporting coverage"
                        sh 'tox -e codecov -- -F regression'
                    } // end steps
                    post {
                        always {
                            junit (
                                allowEmptyResults: true,
                                testResults: '.tmp/py310-reg/reports/*_results.xml'
                            )
                            echo "Deleting Regression Tests workspace ${env.WORKSPACE}"
                            cleanWs()
                            dir("${env.WORKSPACE}@tmp") {
                              deleteDir()
                            }
                        }
                    } // end post
                }
            } // end parallel
        }

        stage('Slower tests') {
            parallel {
                stage('GMOS LS Tests') {
                    agent { label "master" }
                    environment {
                        MPLBACKEND = "agg"
                        DRAGONS_TEST_OUT = "gmosls_tests_outputs"
                        TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                        TMPDIR = "${env.WORKSPACE}/.tmp/gmosls/"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        sh '.jenkins/scripts/setup_dirs.sh'
                        echo "Running tests"
                        sh 'tox -e py310-gmosls -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/gmosls_results.xml ${TOX_ARGS}'
                        echo "Reporting coverage"
                        sh 'tox -e codecov -- -F gmosls'
                    }  // end steps
                    post {
                        always {
                            echo "Running 'archivePlots' from inside GmosArcTests"
                            archiveArtifacts artifacts: "plots/*", allowEmptyArchive: true
                            junit (
                                allowEmptyResults: true,
                                testResults: '.tmp/py310-gmosls/reports/*_results.xml'
                            )
                            echo "Deleting GMOS LS Tests workspace ${env.WORKSPACE}"
                            cleanWs()
                            dir("${env.WORKSPACE}@tmp") {
                              deleteDir()
                            }
                        }  // end always
                    }  // end post
                }  // end stage

                stage('Slow Tests') {
                    agent { label "master" }
                    environment {
                        MPLBACKEND = "agg"
                        DRAGONS_TEST_OUT = "regression_tests_outputs"
                        TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                        TMPDIR = "${env.WORKSPACE}/.tmp/slow/"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        echo "${env.PATH}"
                        sh '.jenkins/scripts/setup_dirs.sh'
                        echo "Slow tests"
                        sh 'tox -e py310-slow -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/slow_results.xml ${TOX_ARGS}'
                        echo "Reporting coverage"
                        sh 'tox -e codecov -- -F slow'
                    } // end steps
                    post {
                        always {
                            junit (
                                allowEmptyResults: true,
                                testResults: '.tmp/py310-slow/reports/*_results.xml'
                            )
                            echo "Deleting GMOS LS Tests workspace ${env.WORKSPACE}"
                            cleanWs()
                            dir("${env.WORKSPACE}@tmp") {
                              deleteDir()
                            }
                        }
                    } // end post
                } // end stage
            } // end parallel
        }

    }
    post {
        success {
            sendNotifications 'SUCCESSFUL'
//            deleteDir() /* clean up our workspace */
        }
        failure {
            sendNotifications 'FAILED'
//            deleteDir() /* clean up our workspace */
        }
        always {
            echo "Delete master workspace ${env.WORKSPACE}"
            cleanWs()
        }
    }
}
 
