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
        timeout(time: 4, unit: 'HOURS')
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
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        sh '.jenkins/scripts/setup_agent.sh'
                        echo "Running tests with Python 3.7"
                        sh 'tox -e py37-unit -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/unittests_results.xml ${TOX_ARGS}'
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
                        failure {
                            echo "Archiving tests results for Unit Tests"
                            sh "find ${DRAGONS_TEST_OUT} -not -name \\*.bz2 -type f -print0 | xargs -0 -n1 -P4 bzip2"
        //                             archiveArtifacts artifacts: "${DRAGONS_TEST_OUT}/**"
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
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        echo "${env.PATH}"
                        sh '.jenkins/scripts/setup_agent.sh'
                        echo "Integration tests"
                        sh 'tox -e py37-integ -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/integration_results.xml ${TOX_ARGS}'
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

                stage('Regression Tests') {
                    agent { label "master" }
                    environment {
                        MPLBACKEND = "agg"
                        PATH = "$JENKINS_CONDA_HOME/bin:$PATH"
                        DRAGONS_TEST_OUT = "regression_tests_outputs"
                        TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                    }
                    steps {
                        echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                        checkout scm
                        echo "${env.PATH}"
                        sh '.jenkins/scripts/setup_agent.sh'
                        echo "Regression tests"
                        sh 'tox -e py37-reg -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/regression_results.xml ${TOX_ARGS}'
                        echo "Reporting coverage"
                        sh 'tox -e codecov -- -F regression'
                    } // end steps
                    post {
                        always {
                            junit (
                                allowEmptyResults: true,
                                testResults: 'reports/*_results.xml'
                            )
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
                        }  // end always
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
            }
            steps {
                echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                checkout scm
                echo "${env.PATH}"
                sh '.jenkins/scripts/setup_agent.sh'
                echo "Slow tests"
                sh 'tox -e py37-slow -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/slow_results.xml ${TOX_ARGS}'
                echo "Reporting coverage"
                sh 'tox -e codecov -- -F slow'
            } // end steps
            post {
                always {
                    junit (
                        allowEmptyResults: true,
                        testResults: 'reports/*_results.xml'
                    )
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
