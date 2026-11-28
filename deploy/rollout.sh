#!/usr/bin/env bash
# Runs on the EC2 instance (started by the Deploy workflow through SSM).
#   rollout.sh <image> <site address> <aws region>
# Logs in to ECR with the instance role, writes .env, pulls the new image and
# restarts the stack, waiting until every container reports healthy.
set -euo pipefail

IMAGE="$1"
SITE="$2"
REGION="$3"
REGISTRY="${IMAGE%%/*}"
cd "$(dirname "$0")/.."

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

if [ ! -f /opt/barahai/grafana-password ]; then
  head -c 18 /dev/urandom | base64 | tr -d '/+=' > /opt/barahai/grafana-password
  chmod 600 /opt/barahai/grafana-password
fi

cat > .env <<EOF
API_IMAGE=${IMAGE}
SITE_ADDRESS=${SITE}
GRAFANA_ADMIN_PASSWORD=$(cat /opt/barahai/grafana-password)
EOF

docker compose --profile monitoring pull --quiet
docker compose --profile monitoring up -d --wait --remove-orphans
docker image prune -af --filter "until=72h" > /dev/null
docker compose ps
