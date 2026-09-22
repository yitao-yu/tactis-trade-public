variable "region" { default = "us-east-2" }
variable "trader_env" { default = "paper" }

variable "vpc_id" {
  description = "Existing VPC; defaults to the account default VPC's first one"
  default = ""
}

variable "subnet_ids" {
  description = "Public subnets for the batch compute env (list). If empty, resolved from default VPC."
  default = []
}

variable "gpu_instance_type" {
  description = "g4dn for NVIDIA T4 (cu118 stack); never g4ad (AMD/ROCm)."
  default = "g4dn.xlarge"
}

variable "job_role_name" { default = "tactis-batch-job-role" }
variable "ecs_role_name" { default = "tactis-batch-ecs-role" }
variable "repo_name" { default = "tactis-live" }
variable "log_group" { default = "/aws/batch/tactis-live" }

variable "ib_sec_pair" {
  description = "SSM SecureString param names for IBKR paper cred"
  type = list(string)
  default = ["/tactis/ibkr_username", "/tactis/ibkr_password"]
}

variable "xq_cookie_param" {
  description = "SSM SecureString param name for the Xueqiu cookie"
  default = "/tactis/xueqiu_cookie"
}
