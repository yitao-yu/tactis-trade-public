# — IAM roles for AWS Batch. THREE distinct trust domains:
#   1. service_role  (CE) — assumed by batch.amazonaws.com to manage instances
#   2. instance_role (CE) — assumed by ec2.amazonaws.com (the launched instances)
#   3. job/execution role (jobdef) — assumed by ecs-tasks.amazonaws.com (the task)

# 1. Batch service role — use the AWS-managed role (already exists, correct trust)
data "aws_iam_role" "batch_service" {
  name = "AWSBatchServiceRole"
}

# 2. Instance role + profile
data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_instance" {
  name               = "tactis-batch-instance-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "batch_ecs" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch" {
  name = "tactis-batch-instance-profile"
  role = aws_iam_role.batch_instance.name
}

# 3. Job role (task assumes it) + execution role (ECR pull + logs)
data "aws_iam_policy_document" "job_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "job" {
  name               = var.job_role_name
  assume_role_policy = data.aws_iam_policy_document.job_assume.json
}

# execution role: ECR pull + CloudWatch logs (task execution)
resource "aws_iam_role" "exec" {
  name               = "tactis-batch-exec-role"
  assume_role_policy = data.aws_iam_policy_document.job_assume.json
}

resource "aws_iam_role_policy_attachment" "exec_managed" {
  role       = aws_iam_role.exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "job_perms" {
  statement {
    sid = "SSMReadTactis"
    actions = [
      "ssm:GetParameter",
      "ssm:GetParameters",
      "ssm:GetParametersByPath",
    ]
    resources = ["arn:aws:ssm:${var.region}:*:parameter/tactis/*"]
  }
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["arn:aws:logs:*:*:*"]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = ["*"]  # tightened once the data bucket exists
  }
}

resource "aws_iam_role_policy" "job" {
  name   = "tactis-job-perms"
  role   = aws_iam_role.job.id
  policy = data.aws_iam_policy_document.job_perms.json
}
