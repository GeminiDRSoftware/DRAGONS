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

        // for kv in mapToList(test_structure) {
        //     run_test_group(kv.key, kv.value, true)
        // }

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
