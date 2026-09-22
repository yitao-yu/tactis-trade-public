# app/batch Terraform

Provider AWS ~>5.0, region us-east-2 (default). Backend local (fine for one operator).

Resources created:
- ECR repo `tactis-live` for the runner image
- IAM roles: Batch ECS instance role + job role (SSM read on /tactis/*, logs, S3)
- CloudWatch log group `/aws/batch/tactis-live` (14d)
- Managed compute env `tactis-gpu-spot`: SPOT, SPOT_CAPACITY_OPTIMIZED,
  g4dn.xlarge (NVIDIA T4 — cu118 stack; do NOT use g4ad/AMD), minvCpus=0,
  maxvCpus=4, public subnets of the default VPC (no NAT Gateway), launch
  template pinned to the Batch ECS GPU AMI SSM param, 60 GB gp3 root
- Job queue `tactis-live-queue`
- Multi-container job def `tactis-live-cycle`:
    container #1 `ib-gateway`  ghnzsnz/ib-gateway (paper, port 4002)
      env from SSM: IB_GATEWAY_USER, IB_GATEWAY_PASSWORD (secretEnvironment)
    container #2 `runner`      ECR tactis-live:latest, /entrypoint.sh
      GPU=1; depends_on ib-gateway (adjust to SUCCESS/HEALTHY per Batch API)
- EventBridge Scheduler `tactis-ibkr-eod`: cron(45 20 ? * MON-FRI *) = 15:45 CT
  weekdays (UTC cron); ECS targets of EventBridge Scheduler need the
  `ecs_parameters` block with the task def ARN + public network config.

Deploy:

    cd app/batch
    terraform init
    terraform plan    # inspect everything, esp. jobdef.tf shapes
    terraform apply

Seed secrets (never via chat):

    aws ssm put-parameter --name /tactis/ibkr_username --type SecureString --tier Standard \
      --value '<ibkr-paper-username>'
    aws ssm put-parameter --name /tactis/ibkr_password --type SecureString --tier Standard \
      --value '<ibkr-paper-password>'
    aws ssm put-parameter --name /tactis/xueqiu_cookie --type SecureString --tier Standard \
      --value '<cookie string>'

Known TODOs before first apply:
  1. schedule.tf: add `depends_on` for the compute env; confirm Every cron
     grammar matches EventBridge Scheduler (cron(…) is fine there).
  2. jobdef.tf: `secretEnvironment` belongs on EACH container (Batch splits
     secrets per container); current draft is top-level — fix on first plan.
  3. Multi-container Batch job defs support depends_on via
     `dependsOn: {containerName, condition}` inside `containers`; verify
     field names in the aws provider docs at apply time.
  4. Docker build in Dockerfile: `IBGateway_USER`/`IBGateway_PASSWORD` env
     names must match what ghcr's image expects — cross-check.
  5. If the IBKR Gateway container needs a license/2FA interplay, see the
     main README "IB Gateway 2FA" note.
  6. Runner ledger path: containers write to ~/.tactis/app-logs — either
     mount a volume or pass run_live.py --log-dir /tmp/log; Batch FSx/efs
     later for tells.

Verified locally (2026-09-20):
- pytest tests/test_app_runner.py: 11/11 pass under py310 micromamba env
  (env `micromamba run -n py310 python -m pytest`).
- docker: tactis-smoke image (Dockerfile.smoke, no torch) runs entrypoint
  end-to-end with repo cfg mounted at /opt/cfg; ledger row appended. TO
  replicate: docker build -f app/Dockerfile.smoke -t tactis-smoke .; docker
  run --rm -v repo/cfg:/opt/cfg:ro -v repo/trading:/opt/trading:ro
  -v weights:/opt/app/weights.json:ro -v /tmp/smokelogs:/root/.tactis/app-logs
  -e TACTIS_VENUE=xueqiu -e TACTIS_WEIGHTS_FILE=/opt/app/weights.json
  -e TACTIS_FORCE_DRYRUN=1 tactis-smoke
