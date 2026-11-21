"""Cloud architecture diagram for BarahAI (diagrams-as-code).

    pip install diagrams          (needs Graphviz: apt install graphviz / winget install Graphviz)
    python docs/architecture.py   -> docs/architecture.png

The "Diagram" GitHub Actions workflow renders it on every change to this file.
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import EC2, ECR
from diagrams.aws.management import SystemsManager
from diagrams.aws.security import IAMRole
from diagrams.aws.storage import S3
from diagrams.onprem.client import Users
from diagrams.onprem.container import Docker
from diagrams.onprem.monitoring import Grafana, Prometheus
from diagrams.onprem.network import Caddy
from diagrams.onprem.vcs import Github
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python

graph = {"fontsize": "22", "pad": "0.6", "nodesep": "0.7", "ranksep": "1.0", "splines": "spline"}
node = {"fontsize": "13"}

with Diagram("BarahAI: training, CI/CD and serving", filename="docs/architecture", show=False,
             direction="LR", graph_attr=graph, node_attr=node):
    players = Users("Players\n(browser)")

    with Cluster("Developer laptop (training)"):
        pipeline = Python("DVC pipeline\ngenerate > relabel > pretrain\n> PPO > evaluate > register")
        mlflow = Python("MLflow\ntracking + model registry\n(@champion, @challenger)")
        pipeline >> Edge(label="params, metrics,\nartifacts") >> mlflow

    with Cluster("GitHub"):
        repo = Github("Repository\ncode, params.yaml,\ndvc.lock")
        ci = Github("CI workflow\nruff, pytest, docker build,\ncompose smoke test")
        cd = Github("Deploy workflow\nOIDC, build, push, roll out")
        repo >> Edge(label="push / PR") >> ci >> Edge(label="on success (main)") >> cd

    with Cluster("AWS ap-southeast-2"):
        dvc_s3 = S3("S3: DVC remote\ndatasets + models/serve")
        oidc = IAMRole("Deploy role\n(GitHub OIDC)")
        registry = ECR("ECR\nbarahai-api images")
        ssm = SystemsManager("SSM Run Command\n(no SSH)")

        with Cluster("EC2 t3.small + Elastic IP  (Docker Compose)"):
            host = EC2("barahai-server")
            caddy = Caddy("Caddy\nHTTPS (sslip.io)")
            api = Fastapi("api\nFastAPI + ONNX Runtime\nchampion + shadow model")
            prom = Prometheus("Prometheus\n/metrics")
            graf = Grafana("Grafana\ndashboards")
            games = Docker("games.db\n(volume)")
            caddy >> api
            api >> games
            prom >> Edge(label="scrape") >> api
            graf >> prom
            caddy >> Edge(label="/grafana") >> graf

    pipeline >> Edge(label="dvc push") >> dvc_s3
    pipeline >> Edge(label="git push") >> repo
    cd >> Edge(label="assume role") >> oidc
    cd >> Edge(label="dvc pull models/serve") >> dvc_s3
    cd >> Edge(label="docker push") >> registry
    cd >> Edge(label="send-command") >> ssm >> Edge(label="rollout.sh") >> host
    host >> Edge(label="docker pull") >> registry
    players >> Edge(label="HTTPS: play, /api, /docs") >> caddy
