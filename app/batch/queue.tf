# — missing resources to make schedule.tf resolvable —
resource "aws_batch_job_queue" "live" {
  name     = "tactis-live-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu_spot.arn
  }
}
