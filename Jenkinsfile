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
        TEST_PATH = "$WORKSPACE/DRAGONS/test_path/"
    }

    stages {
        stage('Checkout') {
            steps {
                echo "Checkout code"
            }
        }
        stage('Build') {
            parallel {
                stage('os1') {
                    stages {
                        stage('build') {
                            steps {
                                echo "build 1"
                            }
                        }
                    }
                }
                stage('os2') {
                    stages {
                        stage('build') {
                            steps {
                                echo "build 1"
                            }
                        }
                    }
                }
                stage('os3') {
                    stages {
                        stage('build') {
                            steps {
                                echo "build 1"
                            }
                        }
                    }
                }
            }
        }
        stage('Test') {
            parallel {
                stage('test build 1') {
                    stages {
                        stage('Pull build 1') {
                            steps {
                                echo "pull build 1"
                            }
                        }
                        stage('test') {
                            steps {
                                echo "test 1"
                            }
                        }
                    }
                }
                stage('test build 2') {
                    stages {
                        stage('Pull build 2') {
                            steps {
                                echo "pull build 1"
                            }
                        }
                        stage('test') {
                            steps {
                                echo "test 2"
                            }
                        }
                    }
                }
                stage('static metrics') {
                    steps {
                        echo "run PyLint and PyDocStyle"
                    }
                }
            }
        }
        stage('Deploy') {
            parallel {
                stage('deploy build 1') {
                    steps {
                        echo "test 1"
                    }
                }
                stage('deploy build 2') {
                    steps {
                        echo "test 1"
                    }
                }
            }
        }
    }
    post {
        failure {
            echo "Send e-mail, when failed"
        }
    }
}

//        stage ("Code pull"){
//            steps{
//                checkout scm
//            }
//        }
//
//        stage ("Set up") {
//            steps {
//                sh  '''
//                    . .jenkins/download_and_install_anaconda.sh
//                    '''
//            }
//        }
//
//        stage('Static Metrics') {
//           steps {
//               echo "PEP8 style check"
//               sh  '''
//                   mkdir -p ./reports
//
//                   pylint --exit-zero --jobs=4 \
//                       astrodata gemini_instruments gempy geminidr \
//                       recipe_system > ./reports/pylint.log
//
//                   pydocstyle --add-ignore D400,D401,D205,D105,D105 \
//                        astrodata gemini_instruments gempy geminidr \
//                        recipe_system > 'reports/pydocstyle.log' || exit 0
//                   '''
//           }
//           post {
//               always {
//                   echo 'Report pyLint warnings using the warnings-ng-plugin'
//                   recordIssues enabledForFailure: true, tool: pyLint(pattern: '**/reports/pylint.log')
//                   echo 'Report pyDocStyle warnings using the warnings-ng-plugin'
//                   recordIssues enabledForFailure: true, tool: pyDocStyle(pattern: '**/reports/pydocstyle.log')
//               }
//           }
//        }



//
//        stage('Deploy') {
//            parallel {
//                stage('deploy_1') {
//                    steps {
//                        echo "deploy 1"
//                    }
//                }
//                stage('deploy_2') {
//                    steps {
//                        echo "deploy 2"
//                    }
//                }
//            }
//        }

//        stage('Unit tests') {X
//            steps {
//                sh  '''
//                    source activate ${BUILD_TAG}
//                    coverage run -m pytest --junit-xml ./reports/test_results.xml
//                    '''
//            }
//            post {
//                always {
//                    echo ' --- Publishing test results --- '
//                    junit (
//                        allowEmptyResults: true,
//                        testResults: 'reports/test_results.xml'
//                        )
//                }
//            }
//        }

//        stage('Code coverage') {
//            steps {
//                sh  '''
//                source activate ${BUILD_TAG}
//                coverage report
//                coverage xml -o ./reports/coverage.xml
//                '''
//            }
//            post {
//                always {
//                    echo ' --- Report coverage usinig Cobertura --- '
//                    step([$class: 'CoberturaPublisher',
//                        autoUpdateHealth: false,
//                        autoUpdateStability: false,
//                        coberturaReportFile: 'reports/coverage.xml',
//                        failNoReports: false,
//                        failUnhealthy: false,
//                        failUnstable: false,
//                        maxNumberOfBuilds: 10,
//                        onlyStable: false,
//                        sourceEncoding: 'ASCII',
//                        zoomCoverageChart: false])
//
//                    echo 'Report on code coverage using Code Coverage API plugin'
//                    publishCoverage adapters: [coberturaAdapter('')]
//                }
//            }
//        }

