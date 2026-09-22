# — Job definition: SINGLE-CONTAINER smoke (test-run stage).
#   Multi-container (ib-gateway + runner) is deferred — see jobdef.multicontainer.tf.disabled
#   for the draft. This smoke image proves CE + queue + ECR + IAM + SSM end-to-end.

resource "aws_batch_job_definition" "live" {
  name = "tactis-live-cycle"
  type = "container"

  container_properties = jsonencode({
    image     = "${aws_ecr_repository.live.repository_url}:smoke"
    jobRoleArn = aws_iam_role.job.arn
    executionRoleArn = aws_iam_role.exec.arn

    command = ["/opt/app/entrypoint.sh"]

    environment = [
      { name = "TACTIS_VENUE",          value = "xueqiu" },
      { name = "TACTIS_FORCE_DRYRUN",   value = "1" },
      { name = "HOME",                  value = "/root" },
      # — inference (checkpoint path) —
      # TACTIS_CHECKPOINT_S3 set => entrypoint downloads ckpt + norm_stats,
      # fetches fresh data, runs infer.py, and overrides the static weights.
      # Swap base <-> RL by editing the filename only (no rebuild needed).
      { name = "TACTIS_CHECKPOINT_S3",  value = "s3://<AWS_ACCOUNT_ID>-tactis-build/checkpoints/base.pth" },
      { name = "TACTIS_NORM_STATS_S3",  value = "s3://<AWS_ACCOUNT_ID>-tactis-build/checkpoints/norm_stats.json" },
      { name = "TACTIS_DATA_DAYS",      value = "40" },
      { name = "TACTIS_NUM_SAMPLES",    value = "64" },
      { name = "TACTIS_MAX_ASSETS",     value = "20" },
    ]

    resourceRequirements = [
      { type = "VCPU",   value = "2" },
      { type = "MEMORY", value = "4096" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = var.log_group
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "smoke"
      }
    }
  })

  retry_strategy {
    attempts = 2
    evaluate_on_exit {
      on_reason = "exit"
      on_status_reason = "Host EC2*"
      action           = "RETRY"
    }
  }
}
