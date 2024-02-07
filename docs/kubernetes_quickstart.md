# Kubernetes Helm Chart for Freqtrade 

This helm chart is like Freqtrade for educational purposes. Any professional use is not recommended. Run Freqtrade in dry-run (see configs).
Please check the logs after deployment for 

```
freqtrade.exchange.exchange - INFO - Instance is running with dry_run enabled
```

## Prerequistes

Look for the appropriate latest version of your preferred OS. Ask ChatGPT/Google to help you.

- basic knowledge about helm, kubernetes and of course Freqtrade
- helm command line tool
- kubectl command line tool 
- minikube
- stern (optional)

This helm chart was tested on minikube for Mac. It should run without any changes on all minikube distribution beginning with version, see below.
On professional cloud platforms like Google GKE more requirements and costs arise like compute engine, firewalls, connectivity, secrets management, etc.  

```
minikube version: v1.32.0
commit: 8220a6eb95f0a4d75f7f2d7b14cef975f050512d
```

## Configuration

To change Freqtrade configuration please refer to these config files

- values-minikube.yaml (holds general freqtrade parameters) 
- values-minikube-creds.yaml (holds api keys (not recommended, see command line and clean history afterwards), and other user specific configs like telegram config, etc.)

## Deployment 

The example uses Kraken as the crypto exchange platform. If you want to use another exchange please change the values in values-minikube-creds.yaml accordingly.  
You can run helm upgrade --install or use the template function like in this example and pipe it to kubectl. The resources get deployed into the default namespace.
This Freqtrade deployment uses free available strategies like Strategy001. By deploying this version you agree to all terms and conditions defined by Freqtrade and all involved tools and services.

```
# Make sure you are in the default k8s namespace 
cd docker/helm-chart
minikube start --vm=true --driver=docker
helm template -f values-base.yaml -f values-minikube.yaml -f values-minikube-creds.yaml --set configcreds.exchange.key="kraken-api-key" --set configcreds.exchange.secret="kraken-api-secret" -n default . | kubectl --context minikube apply -n default -f -
```

## Logging

Use stern to track the log output of your bot by running

```
stern freqtrade -A
```

or use 

```
kubectl logs deploy/freqtrade-default -n default -f
```

## Troubleshooting

In case of errors please check global kubernetes events

```
kubectl get events -A --sort-by='.lastTimestamp'
```

#### Issues found in this version can be reported soon...
