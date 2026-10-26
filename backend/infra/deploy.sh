#!/usr/bin/env bash
# Deploy PillCheck to your AWS account (serverless, pay-per-use).
#
#   cd backend && ./infra/deploy.sh
#
# Nothing is created until you answer "y". Re-running updates the deployment.
# Needs: AWS CLI (configured), Docker Desktop (running). Region: ap-south-1 unless AWS_REGION is set.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
STACK="${PILLCHECK_STACK:-pillcheck}"
REPO="pillcheck"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$HERE/.deploy.env"
cd "$HERE"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ask() { local a; read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[Yy]$ ]]; }
die() { echo "error: $*" >&2; exit 1; }

command -v aws >/dev/null || die "AWS CLI not found (brew install awscli)"
command -v docker >/dev/null || die "Docker not found (install Docker Desktop and start it)"
docker info >/dev/null 2>&1 || die "Docker is not running. Open Docker Desktop and wait until it says 'running'."
[[ -f models/aliked-n16rot-top1k-640.onnx || -f scripts/download_models.py ]] || die "run from the backend folder"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)" || die "AWS CLI is not configured (aws configure)"
CALLER="$(aws sts get-caller-identity --query Arn --output text)"

# ---------------------------------------------------------------- saved settings
touch "$STATE"; chmod 600 "$STATE"
# shellcheck disable=SC1090
source "$STATE"
if [[ -z "${API_TOKEN:-}" ]]; then
  API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  echo "API_TOKEN=$API_TOKEN" >> "$STATE"
fi
if [[ -z "${ALERT_EMAIL+x}" ]]; then
  read -r -p "Email for caregiver alerts (blank = in-app only): " ALERT_EMAIL
  echo "ALERT_EMAIL=$ALERT_EMAIL" >> "$STATE"
fi

# ---------------------------------------------------------------- Lambda concurrency
LIMIT="$(aws lambda get-account-settings --region "$REGION" --query AccountLimit.ConcurrentExecutions --output text)"
if (( LIMIT > 20 )); then RESERVED=2; else RESERVED=0; fi

# ---------------------------------------------------------------- the plan
bold "PillCheck deployment plan"
cat <<EOF
  Account:  $ACCOUNT  ($CALLER)
  Region:   $REGION      Stack: $STACK

  Will create (all pay-per-use; nothing runs while idle):
    - ECR repository '$REPO'          container image storage (~0.6 GB, ~\$0.06/month)
    - Lambda function (arm64, 3 GB)   runs only per request; Free Tier covers this demo
    - Lambda Function URL             public HTTPS endpoint (no load balancer, no fixed cost)
    - DynamoDB table (on-demand)      records; pennies at demo volume
    - S3 bucket (private, encrypted)  photos/evidence; objects expire after 90 days
    - SNS topic                       caregiver alert emails (first 1,000/month free)
    - EventBridge Scheduler           every 15 min reminder check (free allowance)
    - CloudWatch logs (14 days), 1 alarm (~\$0.10/month), 1 dashboard (free tier: 3)
  Spending guards:
    - concurrency: $( ((RESERVED > 0)) && echo "reserved $RESERVED" || echo "account limit $LIMIT (cannot reserve on this account; the account cap applies)" )
    - public demo: per-visitor rate limits, 1,500 photo checks/day overall
  Estimated total for the hackathon (~18 days): about \$1-4, inside your credits.
EOF
echo
ask "Create / update these resources now?" || { echo "Nothing was created."; exit 0; }

# ---------------------------------------------------------------- Bedrock (separate consent)
BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-}"
if [[ -z "$BEDROCK_MODEL_ID" ]]; then
  echo
  bold "Optional: AI planner on Amazon Bedrock (Amazon Nova Pro)"
  echo "  ~\$0.003 per agent decision, capped at 300 decisions/day (~\$1/day worst case)."
  echo "  Without it the agent uses its built-in rule planner (free); everything still works."
  if ask "Enable the Bedrock planner?"; then
    for candidate in apac.amazon.nova-pro-v1:0 amazon.nova-pro-v1:0; do
      if aws bedrock-runtime converse --region "$REGION" --model-id "$candidate" \
          --messages '[{"role":"user","content":[{"text":"Reply with OK"}]}]' \
          --inference-config maxTokens=5 >/dev/null 2>/tmp/pillcheck-bedrock.err; then
        BEDROCK_MODEL_ID="$candidate"; break
      fi
    done
    if [[ -n "$BEDROCK_MODEL_ID" ]]; then
      echo "  Bedrock OK with $BEDROCK_MODEL_ID (test call cost < \$0.0001)."
      echo "BEDROCK_MODEL_ID=$BEDROCK_MODEL_ID" >> "$STATE"
    else
      echo "  Bedrock is not available yet: $(tail -1 /tmp/pillcheck-bedrock.err)"
      echo "  Continuing with the free rule planner. Re-run this script later to enable it."
    fi
  fi
fi

# ---------------------------------------------------------------- image
bold "1/3  Building the container (arm64) and pushing it to ECR"
aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 || {
  aws ecr create-repository --region "$REGION" --repository-name "$REPO" \
    --image-scanning-configuration scanOnPush=true --encryption-configuration encryptionType=AES256 >/dev/null
  aws ecr put-lifecycle-policy --region "$REGION" --repository-name "$REPO" --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep last 3","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":3},"action":{"type":"expire"}}]}' >/dev/null
}
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE="$REGISTRY/$REPO:$TAG"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
[[ -f demo_assets/manifest.json ]] || python3 scripts/make_demo_assets.py
docker buildx build --platform linux/arm64 --provenance=false -f Dockerfile.lambda -t "$IMAGE" --push ..   # context: project root (backend + frontend)

# ---------------------------------------------------------------- stack
bold "2/3  Creating the AWS resources (CloudFormation)"
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
  --template-file infra/template.yaml --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides ImageUri="$IMAGE" ApiToken="$API_TOKEN" AlertEmail="${ALERT_EMAIL:-}" \
      BedrockModelId="$BEDROCK_MODEL_ID" ReservedConcurrency="$RESERVED"

out() { aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
          --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
API_URL="$(out ApiUrl)"; API_URL="${API_URL%/}"
echo "API_URL=$API_URL" > "$STATE.tmp"; grep -v '^API_URL=' "$STATE" >> "$STATE.tmp" || true; mv "$STATE.tmp" "$STATE"; chmod 600 "$STATE"

# ---------------------------------------------------------------- smoke test
bold "3/3  Warming up and checking the deployment"
aws lambda invoke --region "$REGION" --function-name "$STACK" --cli-binary-format raw-in-base64-out \
  --payload '{"pillcheck":"warm"}' /tmp/pillcheck-warm.json >/dev/null && cat /tmp/pillcheck-warm.json; echo
curl -fsS "$API_URL/health" && echo
curl -fsS -o /dev/null -w "demo session: HTTP %{http_code}\n" -X POST -H 'content-type: application/json' -d '{}' "$API_URL/demo/session"

echo
bold "Done."
echo "  App:        $API_URL/        (API docs: $API_URL/docs)"
echo "  Dashboard:  $(out DashboardUrl)"
echo "  Admin token saved in backend/.deploy.env (keep it private)."
[[ -n "${ALERT_EMAIL:-}" ]] && echo "  Check $ALERT_EMAIL for an AWS 'Subscription Confirmation' email and click Confirm."
echo "  Remove everything later with: ./infra/teardown.sh"
