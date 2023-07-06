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

// Change these to automatically skip steps
def runtests_gmosls  = 1  // 1 to enable
def runtests_slow    = 1
def runtests_f2      = 1
def runtests_niri    = 1
def runtests_gsaoi   = 1
def runtests_gnirs   = 1
def runtests_wavecal = 1

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

        stage('F2 Tests') {
            when {
                expression { runtests_f2  == 1 }
            }

            agent { label "master" }
            environment {
                MPLBACKEND = "agg"
                DRAGONS_TEST_OUT = "f2_tests_outputs"
                TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                TMPDIR = "${env.WORKSPACE}/.tmp/f2/"
            }
            steps {
                echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                checkout scm
                sh '.jenkins/scripts/setup_dirs.sh'
                echo "Running tests"
                sh 'tox -e py310-f2 -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/f2_results.xml ${TOX_ARGS}'
                echo "Reporting coverage"
                sh 'tox -e codecov -- -F f2'
            }  // end steps
            post {
                always {
                    echo "Running 'archivePlots' from inside F2 Tests"
                    archiveArtifacts artifacts: "plots/*", allowEmptyArchive: true
                    junit (
                        allowEmptyResults: true,
                        testResults: '.tmp/py310-f2/reports/*_results.xml'
                    )
                    echo "Deleting F2 Tests workspace ${env.WORKSPACE}"
                    cleanWs()
                    dir("${env.WORKSPACE}@tmp") {
                      deleteDir()
                    }
                }  // end always
            }  // end post
        }  // end stage
        stage('GSAOI Tests') {
            when {
                expression { runtests_gsaoi  == 1 }
            }

            agent { label "master" }
            environment {
                MPLBACKEND = "agg"
                DRAGONS_TEST_OUT = "gsaoi_tests_outputs"
                TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                TMPDIR = "${env.WORKSPACE}/.tmp/gsaoi/"
            }
            steps {
                echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                checkout scm
                sh '.jenkins/scripts/setup_dirs.sh'
                echo "Running tests"
                sh 'tox -e py310-gsaoi -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/gsaoi_results.xml ${TOX_ARGS}'
                echo "Reporting coverage"
                sh 'tox -e codecov -- -F gsaoi'
            }  // end steps
            post {
                always {
                    echo "Running 'archivePlots' from inside GSAOI Tests"
                    archiveArtifacts artifacts: "plots/*", allowEmptyArchive: true
                    junit (
                        allowEmptyResults: true,
                        testResults: '.tmp/py310-gsaoi/reports/*_results.xml'
                    )
                    echo "Deleting GSAOI Tests workspace ${env.WORKSPACE}"
                    cleanWs()
                    dir("${env.WORKSPACE}@tmp") {
                      deleteDir()
                    }
                }  // end always
            }  // end post
        }  // end stage
        stage('NIRI Tests') {
            when {
                expression { runtests_niri  == 1 }
            }

            agent { label "master" }
            environment {
                MPLBACKEND = "agg"
                DRAGONS_TEST_OUT = "niri_tests_outputs"
                TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                TMPDIR = "${env.WORKSPACE}/.tmp/niri/"
            }
            steps {
                echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                checkout scm
                sh '.jenkins/scripts/setup_dirs.sh'
                echo "Running tests"
                sh 'tox -e py310-niri -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/niri_results.xml ${TOX_ARGS}'
                echo "Reporting coverage"
                sh 'tox -e codecov -- -F niri'
            }  // end steps
            post {
                always {
                    echo "Running 'archivePlots' from inside NIRI Tests"
                    archiveArtifacts artifacts: "plots/*", allowEmptyArchive: true
                    junit (
                        allowEmptyResults: true,
                        testResults: '.tmp/py310-niri/reports/*_results.xml'
                    )
                    echo "Deleting NIRI Tests workspace ${env.WORKSPACE}"
                    cleanWs()
                    dir("${env.WORKSPACE}@tmp") {
                      deleteDir()
                    }
                }  // end always
            }  // end post
        }  // end stage
        stage('GNIRS Tests') {
            when {
                expression { runtests_gnirs == 1 }
            }

            agent { label "master" }
            environment {
                MPLBACKEND = "agg"
                DRAGONS_TEST_OUT = "gnirs_tests_outputs"
                TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                TMPDIR = "${env.WORKSPACE}/.tmp/gnirs/"
            }
            steps {
                echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                checkout scm
                sh '.jenkins/scripts/setup_dirs.sh'
                echo "Running tests"
                sh 'tox -e py310-gnirs -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/gnirs_results.xml ${TOX_ARGS}'
                echo "Reporting coverage"
                sh 'tox -e codecov -- -F gnirs'
            }  // end steps
            post {
                always {
                    echo "Running 'archivePlots' from inside GNIRS Tests"
                    archiveArtifacts artifacts: "plots/*", allowEmptyArchive: true
                    junit (
                        allowEmptyResults: true,
                        testResults: '.tmp/py310-gnirs/reports/*_results.xml'
                    )
                    echo "Deleting GNIRS Tests workspace ${env.WORKSPACE}"
                    cleanWs()
                    dir("${env.WORKSPACE}@tmp") {
                      deleteDir()
                    }
                }  // end always
            }  // end post
        }  // end stage
        stage('WaveCal Tests') {
            when {
                expression { runtests_wavecal == 1 }
            }

            agent { label "master" }
            environment {
                MPLBACKEND = "agg"
                DRAGONS_TEST_OUT = "wavecal_tests_outputs"
                TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
                TMPDIR = "${env.WORKSPACE}/.tmp/wavecal/"
            }
            steps {
                echo "Running build #${env.BUILD_ID} on ${env.NODE_NAME}"
                checkout scm
                sh '.jenkins/scripts/setup_dirs.sh'
                echo "Running tests"
                sh 'tox -e py310-wavecal -v -- --basetemp=${DRAGONS_TEST_OUT} --junit-xml reports/wavecal_results.xml ${TOX_ARGS}'
                echo "Reporting coverage"
                sh 'tox -e codecov -- -F wavecal'
            }  // end steps
            post {
                always {
                    echo "Running 'archivePlots' from inside WaveCal Tests"
                    archiveArtifacts artifacts: "plots/*", allowEmptyArchive: true
                    junit (
                        allowEmptyResults: true,
                        testResults: '.tmp/py310-wavecal/reports/*_results.xml'
                    )
                    echo "Deleting WaveCal Tests workspace ${env.WORKSPACE}"
                    cleanWs()
                    dir("${env.WORKSPACE}@tmp") {
                      deleteDir()
                    }
                }  // end always
            }  // end post
        }  // end stage

        stage('Slower tests') {
            parallel {
                stage('GMOS LS Tests') {
                    when {
                        expression { runtests_gmosls  == 1 }
                    }

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
                    when {
                        expression { runtests_slow  == 1 }
                    }
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
 
