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

        stage('Code Quality with pyLint') {
            steps {
                sh  '''
                    source activate ${BUILD_TAG}
                    if [ ! -d "reports" ]; then
                        mkdir reports
                        fi

                    pylint --exit-zero --rcfile=gempy/support_files/pylintrc \\
                        astrodata > reports/pylint_astrodata.log
                    pylint --exit-zero --rcfile=gempy/support_files/pylintrc \\
                        gemini_instruments > reports/pylint_gemini_instruments.log
                    pylint --exit-zero --rcfile=gempy/support_files/pylintrc \\
                        geminidr > reports/pylint_geminidr.log
                    pylint --exit-zero --rcfile=gempy/support_files/pylintrc \\
                        gempy > reports/pylint_gempy.log
                    pylint --exit-zero --rcfile=gempy/support_files/pylintrc \\
                        recipe_system > reports/pylint_recipe_system.log
                    '''
            }
            post {
                always {
                    echo 'Report pyLint warnings using the warnings-ng-plugin'
                    recordIssues enabledForFailure: true, tool: pyLint('**/reports/pylint_*.log')
                }
            }
        }

        stage('Verify docstrings') {
            steps {
                sh  '''
                    source activate ${BUILD_TAG}

                    pydocstyle --convention=numpy astrodata > reports/pds_astrodata.log || exit 0
                    pydocstyle --convention=numpy gemini_instruments > reports/pds_gemini_instruments.log || exit 0
                    pydocstyle --convention=numpy geminidr > reports/pds_geminidr.log || exit 0
                    pydocstyle --convention=numpy gempy > reports/pds_gempy.log || exit 0
                    pydocstyle --convention=numpy recipe_system > reports/pds_recipe_system.log || exit 0
                    '''
            }
            post {
                always {
                    echo 'Report pydocstyle using the warnings-ng-plugin'
                    recordIssues enabledForFailure: true, tool: pyDocStyle('**/reports/pds_*.log')
                }
            }
        }

        stage('Python tests and Code Coverage') {
            steps {
                sh  '''
                    source activate ${BUILD_TAG}
                    coverage run -m pytest --junit-xml ./reports/test_results.xml
                    '''
            }
            post {
                always {
                    junit (allowEmptyResults: true,
                        testResults: 'reports/test_results.xml')
                }
            }
        }

        stage('Code coverage report') {
            steps {
                sh  '''
                    source activate ${BUILD_TAG}
                    coverage report
                    coverage xml -o reports/cov_dragons.xml
                    '''
            }
            post {
                always {
                    echo 'Report on code coverage using Cobertura'
                    step ([
                        $class: 'CoberturaPublisher',
                        autoUpdateHealth: false,
                        autoUpdateStability: false,
                        coberturaReportFile: 'reports/cov_report.xml',
                        failNoReports: false,
                        failUnhealthy: false,
                        failUnstable: false,
                        maxNumberOfBuilds: 10,
                        onlyStable: false,
                        sourceEncoding: 'ASCII',
                        zoomCoverageChart: true
                        ])

                    echo 'Report on code coverage using Code Coverage API plugin'
                    publishCoverage adapters: [coberturaAdapter('')]
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
                    archiveArtifacts (
                        allowEmptyArchive: true,
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
}
