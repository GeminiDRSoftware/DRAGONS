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

// @Library('dragons_ci@master') _

def test_structure = ["Quicker tests": ["Unit tests": [unit: "py310-unit"],
                                        "Integration tests": [integration: "py310-integ"],
                                        "Regression tests": [regression: "py310-reg"],
                                        ],
                      "Instrument tests": ["F2 Tests": [f2: "py310-f2"],
                                           "GSAOI Tests": [gsaoi: "py310-gsaoi"],
                                           "NIRI Tests": [niri: "py310-niri"],
                                           "GNIRS Tests": [gnirs: "py310-gnirs"],
                                           ],
                      "WaveCal Tests": [wavecal: "py310-wavecal"],
                      "Slower tests": ["GMOS LS Tests": [gmosls: "py310-gmosls"],
                                       "Slow Tests": [slow: "py310-slow"],
                                       "GHOST Tests": [ghost: "py310-ghost"],
                                       ],
                     ]


def run_test_group(name, group, in_parallel) {
    if (group.size() > 1) {
        if (in_parallel) {
            stage(name) {
                println("PARALLEL ${name}")
                def work = [:]
                group.each { k, v -> work[k] = { run_test_group(k, v, false) } }
                parallel work
            }
        } else {
            stage(name) {
                println("SERIAL ${name}")
                group.each { k, v -> run_test_group(k, v, true) }
            }
        }
    } else {
        // There's only one key/value pair here
        println("Running test group ${name}")
        group.each { k, v -> run_single_test(name, k, v) }
    }

}


def run_single_test(name, mark, environ) {
    println("Running single test ${name} ${mark} ${environ}")
    stage(name) {

        agent{
            label "centos7"
        }
        environment {
            MPLBACKEND = "agg"
            DRAGONS_TEST_OUT = "${mark}_tests_outputs/"
            TOX_ARGS = "astrodata geminidr gemini_instruments gempy recipe_system"
            TMPDIR = "${env.WORKSPACE}/.tmp/${mark}/"
        }

        steps {
            echo "ECHO: Running test ${name} ${mark} ${environ}"
        }

    }
}


pipeline {

    agent any

    //triggers {
    //   //Polls Source Code Manager every 15 mins
    //   pollSCM('*/15 * * * *')
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
                echo "Step would notify STARTED when dragons_ci is available"
                // sendNotifications 'STARTED'
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

        stage('Test suite') {
            steps {
                script {
                    test_structure.each { k, v -> run_test_group(k, v, true) }
                }
            }
        }

    }



    post {
        success {
            echo "Step would notify SUCCESSFUL when dragons_ci is available"
            // sendNotifications 'SUCCESSFUL'
            echo "Step would notify SUCCESSFUL when dragons_ci is available"
            // sendNotifications 'SUCCESSFUL'
//            deleteDir() /* clean up our workspace */
        }
        failure {
            echo "Step would notify FAILED when dragons_ci is available"
            // sendNotifications 'FAILED'
            echo "Step would notify FAILED when dragons_ci is available"
            // sendNotifications 'FAILED'
//            deleteDir() /* clean up our workspace */
        }
        always {
            echo "Delete master workspace ${env.WORKSPACE}"
            cleanWs()
        }
    }
}
