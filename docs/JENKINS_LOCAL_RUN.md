# Local Jenkins CI/CD Runbook

This setup runs Jenkins in a Linux container on Docker Desktop. It is intended for local academic demonstration, not an internet-facing production controller.

## What the pipeline proves

The root `Jenkinsfile` performs:

```text
Checkout
  -> Python validation
  -> Dependency installation
  -> Pytest
  -> Docker image build
  -> Container health smoke test
  -> Optional Docker Hub push
  -> Optional Docker Desktop Kubernetes deployment
```

## 1. Start Jenkins

From the repository root:

```powershell
git pull origin main
docker compose -f jenkins/docker-compose.yml config
docker compose -f jenkins/docker-compose.yml build
docker compose -f jenkins/docker-compose.yml up -d
docker compose -f jenkins/docker-compose.yml ps
```

Jenkins is exposed at:

```text
http://localhost:8081
```

The Compose setup persists Jenkins state in the `jenkins_home` Docker volume and mounts:

- Docker Desktop's Docker socket for image builds and smoke tests
- the current Windows user's `.kube` directory for optional Kubernetes deployment

The controller runs as root only to access the local Docker socket. Do not expose this local setup to the public internet.

## 2. Unlock Jenkins

Retrieve the generated initial password:

```powershell
docker exec voyage-jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

Paste it into the Unlock Jenkins page, choose **Install suggested plugins**, then create the first administrator user.

## 3. Verify tools inside Jenkins

```powershell
docker exec voyage-jenkins git --version
docker exec voyage-jenkins python --version
docker exec voyage-jenkins docker version
docker exec voyage-jenkins kubectl version --client
```

Before enabling the deployment stage, also verify:

```powershell
docker exec voyage-jenkins kubectl get nodes
```

## 4. Create the Pipeline job

In Jenkins:

1. Select **New Item**.
2. Enter `Voyage-Analytics-CICD`.
3. Choose **Pipeline** and select **OK**.
4. Under **Pipeline**, set **Definition** to `Pipeline script from SCM`.
5. Select **Git**.
6. Repository URL:

   ```text
   https://github.com/Sreekar-DS/Voyage-Analytics-Integrating-MLOps-in-Travel.git
   ```

7. Leave credentials empty because the repository is public.
8. Branch specifier: `*/main`.
9. Script path: `Jenkinsfile`.
10. Keep **Lightweight checkout** enabled and save.

## 5. First safe CI run

Run with:

```text
DOCKERHUB_USERNAME = sreekards
PUSH_IMAGE          = false
DEPLOY_K8S          = false
```

This run validates source code, installs dependencies, runs tests, builds the image, and tests `/health` from a sibling container. It does not publish or deploy anything.

## 6. Add Docker Hub credentials

Create a Docker Hub personal access token with Read & Write permission. Store it in Jenkins rather than the repository.

In Jenkins:

1. **Manage Jenkins** -> **Credentials**.
2. Select **System** -> **Global credentials**.
3. Choose **Add Credentials**.
4. Kind: `Username with password`.
5. Username: `sreekards`.
6. Password: the Docker Hub access token.
7. ID: `dockerhub-credentials`.
8. Description: `Docker Hub CI push token`.

Do not use or commit the Docker Hub password directly.

## 7. Push-image run

Run with:

```text
DOCKERHUB_USERNAME = sreekards
PUSH_IMAGE          = true
DEPLOY_K8S          = false
```

Jenkins pushes both:

```text
sreekards/voyage-analytics-api:<JENKINS_BUILD_NUMBER>
sreekards/voyage-analytics-api:latest
```

## 8. Kubernetes deployment run

Only after this succeeds from the Jenkins container:

```powershell
docker exec voyage-jenkins kubectl get nodes
```

run with:

```text
DOCKERHUB_USERNAME = sreekards
PUSH_IMAGE          = true
DEPLOY_K8S          = true
```

The pipeline applies the manifests, changes the Deployment to the immutable Jenkins build-number image, waits for the rollout, and prints the pods.

Verify from the host afterward:

```powershell
kubectl get deployments
kubectl get pods
kubectl get services
kubectl get hpa
Invoke-RestMethod -Uri "http://localhost/health"
```

## 9. Stop and resume Jenkins

Stop while preserving configuration and build history:

```powershell
docker compose -f jenkins/docker-compose.yml down
```

Resume later:

```powershell
docker compose -f jenkins/docker-compose.yml up -d
```

Do not add `--volumes` unless the Jenkins configuration and history should be permanently reset.

## Troubleshooting

### View Jenkins logs

```powershell
docker compose -f jenkins/docker-compose.yml logs --tail=200 jenkins
```

### Open a shell in Jenkins

```powershell
docker exec -it voyage-jenkins bash
```

### Docker command fails inside Jenkins

```powershell
docker exec voyage-jenkins docker version
```

Confirm Docker Desktop is running and the container has `/var/run/docker.sock` mounted.

### Kubernetes is unreachable inside Jenkins

```powershell
docker exec voyage-jenkins cat /root/.kube/config
docker exec voyage-jenkins kubectl config current-context
docker exec voyage-jenkins kubectl get nodes
```

Confirm Docker Desktop Kubernetes is running and the host file `%USERPROFILE%\.kube\config` exists.

### Reset Jenkins completely

This deletes users, credentials, jobs, plugins, and build history:

```powershell
docker compose -f jenkins/docker-compose.yml down --volumes
```
