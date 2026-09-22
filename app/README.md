# app/ — AWS Batch deployment of the live venues (Venue 1 IBKR + Venue 2 Xueqiu)

Architecture: one Batch job per daily cycle = two containers:
`ib-gateway` (ghcr.io/gnzsnz/ib-gateway, paper mode) + `runner` (this repo's
app image). Secrets flow from SSM Parameter Store SecureStrings via the job
definition. EventBridge Scheduler triggers the job (20:45 UTC Mon-Fri for IBKR).

No order leaves dry-run by default. Live enable requires:
  - cfg/application.yaml venue block `live: true`
  - `TACTIS_LIVE=1` in the job env
  - and NOT `TACTIS_TRADING_KILL=1`
(The LiveGuard two-factor gate may never be bypassed.)

## One-time setup

    cd app/batch && terraform init && terraform apply
    # then seed the SSM params (never paste creds in chat):
    aws ssm put-parameter --name /tactis/ibkr_username --type SecureString \
      --value '<ibkr-paper-username>' --tier Standard
    aws ssm put-parameter --name /tactis/ibkr_password --type SecureString \
      --value '<ibkr-paper-password>' --tier Standard
    aws ssm put-parameter --name /tactis/xueqiu_cookie --type SecureString \
      --value '<cookie string>'

## Build + push the runner image

    lab: docker build -t tactis-live .
    docker tag <sha> <ecr-repo-url>:latest && docker push <ecr-repo-url>:latest

## Submit a one-off dry-run (no schedule needed)

    aws batch submit-job --job-name=tactis-test \
      --job-queue=tactis-live-queue --job-definition=tactis-live-cycle

## Xueqiu keepalive

    app/xueqiu/keepalive_lambda.py — deploy as a Python 3.10 Lambda
    triggered daily by EventBridge. On cookie rejection it raises loudly so
    a CloudWatch alarm (sync via alarm.tf, TODO) fires a notification.

## Costs

Expected ~$1–3/mo: g4dn.xlarge spot ~10–30 min/day on batch, minvCpus=0,
public subnets (no NAT Gateway), SSM Standard tier free, ECR ~free.

## Known TODOs / verification steps

  1. smoke: run `run_live.py --venue xueqiu --weights <fixture> --dry-run` locally
  2. smoke: docker-compose with ib-gateway, verify 4002 handshake, then
     `run_live.py --venue ibkr --weights <fixture> --dry-run`
  3. terraform apply and inspect every resource before seeding secrets
  4. first Batch submit in dry-run; watch logs; only then consider live toggles
