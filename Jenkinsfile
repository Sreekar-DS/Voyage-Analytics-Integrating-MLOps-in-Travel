pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    parameters {
        string(name: 'DOCKERHUB_USERNAME', defaultValue: 'YOUR_DOCKERHUB_USERNAME', description: 'Docker Hub username')
        booleanParam(name: 'PUSH_IMAGE', defaultValue: false, description: 'Push the Docker image to Docker Hub')
        booleanParam(name: 'DEPLOY_K8S', defaultValue: false, description: 'Apply Kubernetes manifests after a successful image push')
    }

    environment {
        IMAGE_NAME = 'voyage-analytics-api'
        IMAGE_TAG = "${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Python Validation') {
            steps {
                sh '''
                    python --version
                    python -m compileall api src mlflow_tracking airflow/dags streamlit_app
                '''
            }
        }

        stage('Install Test Dependencies') {
            steps {
                sh '''
                    python -m venv .venv
                    . .venv/bin/activate
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pytest -q tests
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG} .
                    docker tag ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG} ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest
                '''
            }
        }

        stage('Container Smoke Test') {
            steps {
                sh '''
                    docker rm -f voyage-api-smoke >/dev/null 2>&1 || true
                    docker run -d --name voyage-api-smoke -p 5050:5000 ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
                    sleep 8
                    curl --fail http://127.0.0.1:5050/health
                    docker rm -f voyage-api-smoke
                '''
            }
        }

        stage('Push Docker Image') {
            when {
                expression { return params.PUSH_IMAGE }
            }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-credentials',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                        echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
                        docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest
                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to Kubernetes') {
            when {
                allOf {
                    expression { return params.PUSH_IMAGE }
                    expression { return params.DEPLOY_K8S }
                }
            }
            steps {
                sh '''
                    sed "s#YOUR_DOCKERHUB_USERNAME#${DOCKERHUB_USERNAME}#g" kubernetes/deployment.yml | kubectl apply -f -
                    kubectl apply -f kubernetes/service.yml
                    kubectl apply -f kubernetes/hpa.yml
                    kubectl rollout status deployment/voyage-analytics-api --timeout=180s
                '''
            }
        }
    }

    post {
        always {
            sh 'docker rm -f voyage-api-smoke >/dev/null 2>&1 || true'
            archiveArtifacts artifacts: 'kubernetes/*.yml, Dockerfile, Jenkinsfile', allowEmptyArchive: true
        }
        success {
            echo 'Voyage Analytics CI/CD pipeline completed successfully.'
        }
        failure {
            echo 'Voyage Analytics CI/CD pipeline failed. Review the stage logs before redeploying.'
        }
    }
}
