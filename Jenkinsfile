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

                    else
                        echo "Anaconda is already installed --- Skipping step."
                    fi

                    conda config --add channels http://ssb.stsci.edu/astroconda
                    conda config --add channels http://astroconda.gemini.edu/public/noarch
                    conda update --quiet conda
                    '''
            }
        }

        stage ("Build and Test Environment") {
            steps {
                sh  'conda env create --quiet --file .jenkins/conda_venv.yml -n ${BUILD_TAG}'
                sh  '''
                    source activate ${BUILD_TAG}
                    .jenkins/test_env_and_install_missing_libs.sh
                    '''
                sh  'python .jenkins/download_test_data.py'
            }
        }

        stage('Static code metrics') {
            steps {
                echo "PEP8 style check"
                sh  'mkdir -p ./reports'
                sh  '''
                    source activate ${BUILD_TAG}

                    pylint --exit-zero --jobs=4 \
                        --rcfile=gempy/support_files/pylintrc \
                        astrodata gemini_instruments gempy geminidr \
                        recipe_system > ./reports/pylint.log
                    '''
            }
            post {
                always {
                    echo 'Report pyLint warnings using the warnings-ng-plugin'
                    recordIssues enabledForFailure: true, tool: pyLint(pattern: '**/reports/pylint.log')
                }
            }
        }

        stage('Checking docstrings') {
            steps {
                sh  '''
                    source activate ${BUILD_TAG}
                    pydocstyle --add-ignore D400,D401,D205,D105,D105 \
                        astrodata gemini_instruments gempy geminidr \
                        recipe_system > 'reports/pydocstyle.log' || exit 0
                    '''
            }
            post {
                always {
                    echo 'Report pyDocStyle warnings using the warnings-ng-plugin'
                    recordIssues enabledForFailure: true, tool: pyDocStyle(pattern: '**/reports/pydocstyle.log')
                }
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
                success {
                    echo ' --- Report coverage usinig Cobertura --- '
                    step(
                      [
                        $class: 'CoberturaPublisher',
                        autoUpdateHealth: false,
                        autoUpdateStability: false,
                        coberturaReportFile: 'reports/coverage.xml',
                        failNoReports: false,
                        failUnhealthy: false,
                        failUnstable: false,
                        maxNumberOfBuilds: 10,
                        onlyStable: false,
                        sourceEncoding: 'ASCII',
                        zoomCoverageChart: false
                      ]
                    )

                    // echo 'Report on code coverage using Code Coverage API plugin'
                    // publishCoverage adapters: [coberturaAdapter('')]
                }
            }
        }
    }
    post {
        always {
            sh 'conda remove --name ${BUILD_TAG} --all --quiet --yes'
        }
    }
}
