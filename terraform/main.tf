# ==============================================================================
# Terraform Configuration for Asteroid Detection Platform
# ==============================================================================

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }
}

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "functions_source_dir" {
  description = "Path to Cloud Functions source code"
  type        = string
  default     = "../functions"
}

# ------------------------------------------------------------------------------
# Providers
# ------------------------------------------------------------------------------

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ------------------------------------------------------------------------------
# Enable Required APIs
# ------------------------------------------------------------------------------

resource "google_project_service" "apis" {
  for_each = toset([
    "cloudfunctions.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "firestore.googleapis.com",
    "firebase.googleapis.com",
    "identitytoolkit.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudtasks.googleapis.com",
    "secretmanager.googleapis.com"
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ------------------------------------------------------------------------------
# Cloud Storage Buckets
# ------------------------------------------------------------------------------

resource "google_storage_bucket" "uploads" {
  name          = "${var.project_id}-asteroid-uploads-${var.environment}"
  location      = var.region
  force_destroy = var.environment != "prod"

  uniform_bucket_level_access = true

  cors {
    origin          = ["*"]
    method          = ["GET", "PUT", "POST", "DELETE", "OPTIONS"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    app         = "asteroid-detection"
  }

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "results" {
  name          = "${var.project_id}-asteroid-results-${var.environment}"
  location      = var.region
  force_destroy = var.environment != "prod"

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    app         = "asteroid-detection"
  }

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "functions_source" {
  name          = "${var.project_id}-functions-source-${var.environment}"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  depends_on = [google_project_service.apis]
}

# ------------------------------------------------------------------------------
# Firestore Database
# ------------------------------------------------------------------------------

resource "google_firestore_database" "default" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  depends_on = [google_project_service.apis]
}

# ------------------------------------------------------------------------------
# Service Account
# ------------------------------------------------------------------------------

resource "google_service_account" "functions_sa" {
  account_id   = "asteroid-detection-fn-${var.environment}"
  display_name = "Asteroid Detection Cloud Functions"
  project      = var.project_id
}

resource "google_project_iam_member" "functions_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.functions_sa.email}"
}

resource "google_project_iam_member" "functions_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.functions_sa.email}"
}

resource "google_project_iam_member" "functions_firebase_auth" {
  project = var.project_id
  role    = "roles/firebaseauth.viewer"
  member  = "serviceAccount:${google_service_account.functions_sa.email}"
}

# ------------------------------------------------------------------------------
# Cloud Functions Source
# ------------------------------------------------------------------------------

data "archive_file" "functions_source" {
  type        = "zip"
  source_dir  = var.functions_source_dir
  output_path = "${path.module}/functions-source.zip"
}

resource "google_storage_bucket_object" "functions_source" {
  name   = "functions-source-${data.archive_file.functions_source.output_md5}.zip"
  bucket = google_storage_bucket.functions_source.name
  source = data.archive_file.functions_source.output_path
}

# ------------------------------------------------------------------------------
# Cloud Functions (2nd Gen)
# ------------------------------------------------------------------------------

resource "google_cloudfunctions2_function" "health_check" {
  name     = "asteroid-health-${var.environment}"
  location = var.region
  project  = var.project_id

  build_config {
    runtime     = "python311"
    entry_point = "health_check"
    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.functions_source.name
      }
    }
  }

  service_config {
    max_instance_count    = 10
    available_memory      = "256M"
    timeout_seconds       = 60
    service_account_email = google_service_account.functions_sa.email
    environment_variables = {
      STORAGE_BUCKET = google_storage_bucket.uploads.name
      ENVIRONMENT    = var.environment
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloudfunctions2_function" "create_job" {
  name     = "asteroid-create-job-${var.environment}"
  location = var.region
  project  = var.project_id

  build_config {
    runtime     = "python311"
    entry_point = "create_detection_job"
    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.functions_source.name
      }
    }
  }

  service_config {
    max_instance_count    = 100
    available_memory      = "512M"
    timeout_seconds       = 120
    service_account_email = google_service_account.functions_sa.email
    environment_variables = {
      STORAGE_BUCKET = google_storage_bucket.uploads.name
      ENVIRONMENT    = var.environment
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloudfunctions2_function" "process_job" {
  name     = "asteroid-process-job-${var.environment}"
  location = var.region
  project  = var.project_id

  build_config {
    runtime     = "python311"
    entry_point = "process_detection_job"
    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.functions_source.name
      }
    }
  }

  service_config {
    max_instance_count    = 50
    available_memory      = "4Gi"
    available_cpu         = "2"
    timeout_seconds       = 540
    service_account_email = google_service_account.functions_sa.email
    environment_variables = {
      STORAGE_BUCKET = google_storage_bucket.uploads.name
      RESULTS_BUCKET = google_storage_bucket.results.name
      ENVIRONMENT    = var.environment
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloudfunctions2_function" "get_status" {
  name     = "asteroid-get-status-${var.environment}"
  location = var.region
  project  = var.project_id

  build_config {
    runtime     = "python311"
    entry_point = "get_job_status"
    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.functions_source.name
      }
    }
  }

  service_config {
    max_instance_count    = 100
    available_memory      = "256M"
    timeout_seconds       = 60
    service_account_email = google_service_account.functions_sa.email
    environment_variables = {
      STORAGE_BUCKET = google_storage_bucket.uploads.name
      ENVIRONMENT    = var.environment
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloudfunctions2_function" "upload_url" {
  name     = "asteroid-upload-url-${var.environment}"
  location = var.region
  project  = var.project_id

  build_config {
    runtime     = "python311"
    entry_point = "get_signed_upload_url"
    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.functions_source.name
      }
    }
  }

  service_config {
    max_instance_count    = 100
    available_memory      = "256M"
    timeout_seconds       = 60
    service_account_email = google_service_account.functions_sa.email
    environment_variables = {
      STORAGE_BUCKET = google_storage_bucket.uploads.name
      ENVIRONMENT    = var.environment
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloudfunctions2_function" "list_jobs" {
  name     = "asteroid-list-jobs-${var.environment}"
  location = var.region
  project  = var.project_id

  build_config {
    runtime     = "python311"
    entry_point = "list_user_jobs"
    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.functions_source.name
      }
    }
  }

  service_config {
    max_instance_count    = 100
    available_memory      = "256M"
    timeout_seconds       = 60
    service_account_email = google_service_account.functions_sa.email
    environment_variables = {
      STORAGE_BUCKET = google_storage_bucket.uploads.name
      ENVIRONMENT    = var.environment
    }
  }

  depends_on = [google_project_service.apis]
}

# ------------------------------------------------------------------------------
# IAM - Allow unauthenticated access to Cloud Functions
# ------------------------------------------------------------------------------

resource "google_cloud_run_service_iam_member" "health_invoker" {
  project  = var.project_id
  location = var.region
  service  = google_cloudfunctions2_function.health_check.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "create_job_invoker" {
  project  = var.project_id
  location = var.region
  service  = google_cloudfunctions2_function.create_job.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "process_job_invoker" {
  project  = var.project_id
  location = var.region
  service  = google_cloudfunctions2_function.process_job.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "get_status_invoker" {
  project  = var.project_id
  location = var.region
  service  = google_cloudfunctions2_function.get_status.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "upload_url_invoker" {
  project  = var.project_id
  location = var.region
  service  = google_cloudfunctions2_function.upload_url.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "list_jobs_invoker" {
  project  = var.project_id
  location = var.region
  service  = google_cloudfunctions2_function.list_jobs.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ------------------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------------------

output "uploads_bucket" {
  value       = google_storage_bucket.uploads.name
  description = "Name of the uploads bucket"
}

output "results_bucket" {
  value       = google_storage_bucket.results.name
  description = "Name of the results bucket"
}

output "health_check_url" {
  value       = google_cloudfunctions2_function.health_check.service_config[0].uri
  description = "URL for health check endpoint"
}

output "create_job_url" {
  value       = google_cloudfunctions2_function.create_job.service_config[0].uri
  description = "URL for create job endpoint"
}

output "process_job_url" {
  value       = google_cloudfunctions2_function.process_job.service_config[0].uri
  description = "URL for process job endpoint"
}

output "get_status_url" {
  value       = google_cloudfunctions2_function.get_status.service_config[0].uri
  description = "URL for get status endpoint"
}

output "upload_url_endpoint" {
  value       = google_cloudfunctions2_function.upload_url.service_config[0].uri
  description = "URL for upload URL endpoint"
}

output "list_jobs_url" {
  value       = google_cloudfunctions2_function.list_jobs.service_config[0].uri
  description = "URL for list jobs endpoint"
}

output "service_account_email" {
  value       = google_service_account.functions_sa.email
  description = "Service account email for Cloud Functions"
}
