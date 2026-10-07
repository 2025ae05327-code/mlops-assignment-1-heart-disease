# Kubernetes Deployment

The manifests target local Minikube and deploy 10 resources in the `heart-disease`
namespace. Build the model and image inside Minikube's Docker daemon before applying them.

```bash
minikube start --driver=docker --cpus=2 --memory=4096
eval "$(minikube docker-env)"
docker build -t assignment1-heart-api:1.0.0 .
kubectl apply -k k8s
kubectl rollout status deployment/heart-api -n heart-disease --timeout=180s
```

For a cloud cluster, push the image to its registry, change the API Deployment `image`, and
change `imagePullPolicy: Never` to `IfNotPresent`. Keep the probes, resources, and service.
Use managed Prometheus/Grafana and a restricted secret for Grafana in a real environment.
