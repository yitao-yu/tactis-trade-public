# — launch template with NVIDIA driver / ECS GPU AMI + security group —

data "aws_ssm_parameter" "batch_gpu_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id"
}

resource "aws_launch_template" "gpu_ami" {
  name = "tactis-gpu-ami"
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 60
      volume_type = "gp3"
      iops        = 3000
      throughput  = 125
      delete_on_termination = true
    }
  }
}

resource "aws_security_group" "batch" {
  name   = "tactis-batch-nodes"
  vpc_id = data.aws_vpc.default.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}