# — ECR repo + image build/push (runner image; Gateway runs from the public
#   gnzsnz/ib-gateway image — see job-def.json) —

resource "aws_ecr_repository" "live" {
  name = var.repo_name
  image_scanning_configuration { scan_on_push = true }
}

# — CloudWatch —
resource "aws_cloudwatch_log_group" "batch" {
  name              = var.log_group
  retention_in_days = 14
}

# — Batch compute environment: g4dn.xlarge spot, minvCpus 0 —
resource "aws_batch_compute_environment" "gpu_spot" {
  compute_environment_name = "tactis-gpu-spot-v2"
  service_role        = data.aws_iam_role.batch_service.arn
  type                = "MANAGED"
  state               = "ENABLED"

  compute_resources {
    type                   = "SPOT"
    allocation_strategy    = "SPOT_CAPACITY_OPTIMIZED"
    instance_type          = [var.gpu_instance_type]
    min_vcpus              = 0
    desired_vcpus          = 0
    max_vcpus              = 4
    security_group_ids     = [aws_security_group.batch.id]
    subnets                = length(var.subnet_ids) > 0 ? var.subnet_ids : data.aws_subnets.public.ids
    instance_role          = aws_iam_instance_profile.batch.arn

    launch_template {
      launch_template_id = aws_launch_template.gpu_ami.id
      version            = "$Latest"
    }

    ec2_configuration {
      image_id_override = data.aws_ssm_parameter.batch_gpu_ami.value  # filled below
      image_type        = "ECS_AL2_NVIDIA"
    }
  }
}
