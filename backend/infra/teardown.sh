#!/usr/bin/env bash
# Delete every AWS resource PillCheck created (stack, bucket contents, container images).
#   cd backend && ./infra/teardown.sh
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
STACK="${PILLCHECK_STACK:-pillcheck}"
REPO="pillcheck"

echo "This permanently deletes the '$STACK' stack in $REGION: the API, all records, all photos,"
echo "and the '$REPO' container repository. It cannot be undone."
read -r -p "Type 'delete' to continue: " answer
[[ "$answer" == "delete" ]] || { echo "Cancelled. Nothing was deleted."; exit 0; }

BUCKET="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text 2>/dev/null || true)"
if [[ -n "$BUCKET" && "$BUCKET" != "None" ]]; then
  echo "Emptying bucket $BUCKET ..."
  aws s3 rm "s3://$BUCKET" --recursive --region "$REGION" >/dev/null
fi

echo "Deleting stack $STACK ..."
aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK"

if aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1; then
  echo "Deleting container repository $REPO ..."
  aws ecr delete-repository --region "$REGION" --repository-name "$REPO" --force >/dev/null
fi

echo "All PillCheck resources are deleted. Nothing further will be billed."
