pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    parameters {
        string(name: 'DOCKERHUB_USERNAME', defaultValue: 'sreekards', description: 'Docker Hub username')
        booleanParam(name: 'PUSH_IMAGE', defaultValue: false, description: 'Push the Docker image to Docker Hub')
        booleanParam(name: 'DEPLOY_K8S', defaultValue: false, description: 'Deploy the pushed build-number image to Docker Desktop Kubernetes')
    }

    environment {
        IMAGE_NAME = 'voyage-analytics-api'
        IMAGE_TAG = "${BUILD_NUMBER}"
        SMOKE_CONTAINER = 'voyage-api-smoke'
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
                    python -m compileall -q api src mlflow_tracking airflow/dags streamlit_app tests
                '''
            }
        }

        stage('Install Test Dependencies') {
            steps {
                sh '''
                    rm -rf .venv
                    python -m venv .venv
                    . .venv/bin/activate
                    python -m pip install --upgrade pip
                    python -m pip install -r requirements.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python -m pytest -q tests
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
                    docker rm -f ${SMOKE_CONTAINER} >/dev/null 2>&1 || true
                    JENKINS_NETWORK=$(docker inspect -f '{{range $name, $_ := .NetworkSettings.Networks}}{{$name}}{{end}}' "$(hostname)")
                    docker run -d \
                        --name ${SMOKE_CONTAINER} \
                        --network "$JENKINS_NETWORK" \
                        ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}

                    for attempt in $(seq 1 30); do
                        if curl --fail --silent --show-error http://${SMOKE_CONTAINER}:5000/health; then
                            docker rm -f ${SMOKE_CONTAINER}
                            exit 0
                        fi
                        sleep 2
                    done

                    docker logs ${SMOKE_CONTAINER}
                    docker rm -f ${SMOKE_CONTAINER}
                    exit 1
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
                    kubectl get nodes
                    kubectl apply -f kubernetes/deployment.yml
                    kubectl set image deployment/voyage-analytics-api \
                        voyage-analytics-api=${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
                    kubectl apply -f kubernetes/service.yml
                    kubectl apply -f kubernetes/hpa.yml
                    kubectl rollout status deployment/voyage-analytics-api --timeout=180s
                    kubectl get pods
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
