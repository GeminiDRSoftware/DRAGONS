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

pipeline {

    agent any

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
        DRAGONS_TEST_INPUTS = "$JENKINS_HOME/dragons_tests/inputs/"
        DRAGONS_TEST_OUTPUTS = "$WORKSPACE/dragons_tests/outputs/"
        DRAGONS_TEST_REFERENCES = "$WORKSPACE/dragons_tests/refs/"
    }

    stages {

        stage ("Prepare"){

            steps{
                checkout scm
                sh '.jenkins/scripts/download_and_install_anaconda.sh'
                sh '.jenkins/scripts/create_conda_environment.sh'
                sh '.jenkins/scripts/install_missing_packages.sh'
                sh '.jenkins/scripts/install_dragons.sh'
                sh '''source activate ${CONDA_ENV_NAME}
                      python .jenkins/scripts/download_test_data.py
                      '''
                sh '.jenkins/scripts/test_environment.sh'
                sh 'rm -rf ./reports'
                sh 'mkdir -p ./reports'
            }

        }

        stage('Code Metrics') {

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
                sh  '''
                    source activate ${CONDA_ENV_NAME}
                    coverage run -m pytest --junit-xml ./reports/test_results.xml
                    '''
                sh  '''
                    source activate ${CONDA_ENV_NAME}
                    python -m coverage xml -o ./reports/coverage.xml
                    '''
            }
            post {
                always {
                    junit (
                        allowEmptyResults: true,
                        testResults: 'reports/test_results.xml'
                        )
                }
            }
        }

        stage('Integration tests') {
            steps {
                echo 'No integration tests defined yet'
            }
        }

        stage('Pack and deliver') {
            steps {
                echo 'Add a step here for packing DRAGONS into a tarball'
                echo 'Make tarball available'
            }
        }

    }
    //post {
    //    always {
    //        sh 'conda remove --name ${CONDA_ENV_NAME} --all --quiet --yes'
    //    }
    //}
}
