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
        label 'bquint-ld1'
    }

    triggers {
        pollSCM('H * * * *')  // Polls Source Code Manager every hour
    }

    options {
        skipDefaultCheckout(true)
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }

    environment {
        PATH = "$JENKINS_HOME/anaconda3/bin:$PATH"
        CONDA_ENV_FILE = ".jenkins/conda_py3env_stable.yml"
        CONDA_ENV_NAME = "py3_stable"
        PYTEST_ARGS = "--remote-data=any --basetemp=/data/jenkins/dragons/outputs"
    }

    stages {

        stage ("Prepare"){

            steps{
                sendNotifications 'STARTED'
                checkout scm
                sh 'git clean -fxd'
                sh 'mkdir plots reports'
                // sh 'rm -rf ./plots; mkdir -p ./plots'
                // sh 'rm -rf ./reports; mkdir -p ./reports'
                sh '.jenkins/scripts/download_and_install_anaconda.sh'
                // sh '.jenkins/scripts/create_conda_environment.sh'
                // sh '.jenkins/scripts/install_missing_packages.sh'
                // sh '.jenkins/scripts/install_dragons.sh'
                // sh '''source activate ${CONDA_ENV_NAME}
                //       python .jenkins/scripts/download_test_inputs.py .jenkins/test_files.txt || echo 0
                //       '''
                // sh '.jenkins/scripts/test_environment.sh'
                // sh 'conda list -n ${CONDA_ENV_NAME}'
            }

        }

        stage('Code Metrics') {

            when {
                branch 'master'
            }
            steps {
                sh '.jenkins/code_metrics/pylint.sh'
                sh '.jenkins/code_metrics/pydocstring.sh'
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

        stage('Unit tests') {

            steps {
                // echo "ensure cleaning __pycache__"
                // sh  'find . | grep -E "(__pycache__|\\.pyc|\\.pyo$)" | xargs rm -rfv'

                echo "Running tests"
                sh 'tox -e py36-unit -v -- --junit-xml ./reports/unittests_results.xml'
                // sh  '''
                //     source activate ${CONDA_ENV_NAME}
                //     coverage run -m pytest ${PYTEST_ARGS} -m "not integtest and not gmosls" --junit-xml ./reports/unittests_results.xml
                //     '''
            }

        }

        stage('GMOS LS Tests') {

            steps {
                echo "Running tests"
                sh 'tox -e py36-gmosls -v -- --junit-xml ./reports/unittests_results.xml'
                // sh  '''
                //     source activate ${CONDA_ENV_NAME}
                //     coverage run -m pytest ${PYTEST_ARGS} -m gmosls --junit-xml ./reports/gmoslstests_results.xml
                //     '''

                // echo "Reporting coverage"
                // sh  '''
                //     source activate ${CONDA_ENV_NAME}
                //     python -m coverage xml -o ./reports/coverage.xml
                //     '''
            }
            post {
                always {
                    echo "Running 'archivePlots' from inside GmosArcTests"
                    archiveArtifacts artifacts: "plots/*", allowEmptyArchive: true
                }
            }

        }

        stage('Integration tests') {

            // when {
            //     branch 'master'
            // }
            steps {
                echo "Integration tests"
                sh 'tox -e py36-integ -v -- --junit-xml ./reports/integration_results.xml'
                // sh  '''
                //     source activate ${CONDA_ENV_NAME}
                //     coverage run -m pytest ${PYTEST_ARGS} -m integtest --junit-xml ./reports/integration_results.xml
                //     '''
            }

        }


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
