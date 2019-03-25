#!/usr/bin/env groovy
pipeline {

    agent any

    triggers {
        pollSCM('H/20 * * * 1-5')
    }

    options {
        skipDefaultCheckout(true)
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timestamps()
    }

    environment {
        PATH = "$JENKINS_HOME/anaconda3/bin:$PATH"
        TEST_PATH = "$WORKSPACE/test_path/"
    }

    stages {

        stage ("Code pull"){
            steps{
                checkout scm
            }
        }

        stage ("Download and Install Anaconda") {
            steps {
                sh  '''
                    if ! [ "$(command -v conda)" ]; then
                        echo "Conda is not installed - Downloading and installing"

                        curl https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh \\
                        --output anaconda.sh --silent

                        chmod a+x anaconda.sh
                        ./anaconda.sh -u -b -p $JENKINS_HOME/anaconda3/

                        conda config --add channels http://ssb.stsci.edu/astroconda
                        conda update --quiet conda
                    else
                        echo "Anaconda is already installed --- Skipping step."
                    fi
              '''
            }
        }

        stage ("Build and Test Environment") {
            steps {
                sh  '''
                    conda env create --quiet --file .jenkins/conda_venv.yml -n ${BUILD_TAG}
                    source activate ${BUILD_TAG}

                    .jenkins/test_env_and_install_missing_libs.sh
                    python .jenkins/download_test_data.py
                    '''
            }
        }

        stage('Static code metrics') {
            steps {
                echo "PEP8 style check"
                sh  '''
                    source activate ${BUILD_TAG}
                    pylint --exit-zero --jobs=4 --rcfile=.pylintrc \
                        astrodata gemini_instruments gempy geminidr \
                        recipe_system > reports>pylint.log
                    '''
            }
        }

        stage('Unit tests') {
            steps {
                sh  '''
                    source activate ${BUILD_TAG}
                    coverage run -m pytest --junit-xml ./reports/test_results.xml
                    '''
            }
            post {
                always {
                    echo ' --- Publishing test results --- '
                    junit (
                        allowEmptyResults: true,
                        testResults: 'reports/test_results.xml'
                        )
                }
            }
        }

        stage('Code coverage') {
            steps {
                sh  '''
                source activate ${BUILD_TAG}
                coverage report
                coverage xml -o ./reports/coverage.xml
                '''
            }
            post {
                always {
                    echo ' --- Report coverage usinig Cobertura --- '
                    step([$class: 'CoberturaPublisher',
                        autoUpdateHealth: false,
                        autoUpdateStability: false,
                        coberturaReportFile: 'reports/coverage.xml',
                        failNoReports: false,
                        failUnhealthy: false,
                        failUnstable: false,
                        maxNumberOfBuilds: 10,
                        onlyStable: false,
                        sourceEncoding: 'ASCII',
                        zoomCoverageChart: false])
                }
            }
        }

        stage('Build package') {
            when {
                expression {
                    currentBuild.result == null || currentBuild.result == 'SUCCESS'
                }
            }
            steps {
                sh  '''
                source activate ${BUILD_TAG}
                python setup.py sdist bdist_egg
                '''
            }
            post {
                always {
                    // Archive unit tests for the future
                    archiveArtifacts (allowEmptyArchive: true,
                        artifacts: 'dist/*whl',
                        fingerprint: true)
                }
            }
        }
    }
    post {
        always {
            sh 'conda remove --yes --all --quiet -n ${BUILD_TAG}'
        }
        failure {
            echo "Send e-mail, when failed"
        }
    }
} // pipeline
