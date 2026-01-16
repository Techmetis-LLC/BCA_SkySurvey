#!/bin/bash
# ==============================================================================
# Asteroid Detection Platform - Deployment Script
# ==============================================================================
# Deploys infrastructure to Google Cloud Platform using Terraform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
PROJECT_ID=""
REGION="us-central1"
ENVIRONMENT="dev"
SKIP_TERRAFORM=false
SKIP_FRONTEND=false

# Parse arguments
usage() {
    echo "Usage: $0 -p PROJECT_ID [-r REGION] [-e ENVIRONMENT] [--skip-terraform] [--skip-frontend]"
    echo ""
    echo "Options:"
    echo "  -p PROJECT_ID     GCP Project ID (required)"
    echo "  -r REGION         GCP Region (default: us-central1)"
    echo "  -e ENVIRONMENT    Environment: dev, staging, prod (default: dev)"
    echo "  --skip-terraform  Skip Terraform deployment"
    echo "  --skip-frontend   Skip frontend deployment"
    echo "  -h                Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -p) PROJECT_ID="$2"; shift 2 ;;
        -r) REGION="$2"; shift 2 ;;
        -e) ENVIRONMENT="$2"; shift 2 ;;
        --skip-terraform) SKIP_TERRAFORM=true; shift ;;
        --skip-frontend) SKIP_FRONTEND=true; shift ;;
        -h) usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

# Validate required arguments
if [[ -z "$PROJECT_ID" ]]; then
    log_error "Project ID is required"
    usage
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info "=============================================="
log_info "Asteroid Detection Platform Deployment"
log_info "=============================================="
log_info "Project ID:  $PROJECT_ID"
log_info "Region:      $REGION"
log_info "Environment: $ENVIRONMENT"
log_info "=============================================="

# Check prerequisites
log_info "Checking prerequisites..."

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is required but not installed"
        exit 1
    fi
    log_success "$1 found"
}

check_command gcloud
check_command terraform

if [[ "$SKIP_FRONTEND" == false ]]; then
    check_command npm
    check_command firebase
fi

# Set GCP project
log_info "Setting GCP project..."
gcloud config set project "$PROJECT_ID"

# Enable required APIs
log_info "Enabling required APIs..."
APIS=(
    "cloudfunctions.googleapis.com"
    "cloudbuild.googleapis.com"
    "storage.googleapis.com"
    "firestore.googleapis.com"
    "firebase.googleapis.com"
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
)

for api in "${APIS[@]}"; do
    gcloud services enable "$api" --quiet
done
log_success "APIs enabled"

# Copy detector module to functions directory
log_info "Preparing Cloud Functions..."
cp "$PROJECT_ROOT/src/asteroid_detector.py" "$PROJECT_ROOT/functions/"
log_success "Detector module copied to functions directory"

# Deploy Terraform infrastructure
if [[ "$SKIP_TERRAFORM" == false ]]; then
    log_info "Deploying infrastructure with Terraform..."
    cd "$PROJECT_ROOT/terraform"
    
    # Create tfvars file
    cat > terraform.tfvars << EOF
project_id = "$PROJECT_ID"
region = "$REGION"
environment = "$ENVIRONMENT"
EOF
    
    # Initialize Terraform
    terraform init
    
    # Plan and apply
    terraform plan -out=tfplan
    terraform apply tfplan
    
    # Get outputs
    HEALTH_URL=$(terraform output -raw health_check_url)
    CREATE_JOB_URL=$(terraform output -raw create_job_url)
    PROCESS_JOB_URL=$(terraform output -raw process_job_url)
    GET_STATUS_URL=$(terraform output -raw get_status_url)
    UPLOAD_URL=$(terraform output -raw upload_url_endpoint)
    LIST_JOBS_URL=$(terraform output -raw list_jobs_url)
    
    log_success "Infrastructure deployed!"
    
    # Save URLs to file for frontend
    cat > "$PROJECT_ROOT/frontend/.env.local" << EOF
REACT_APP_API_URL=$CREATE_JOB_URL
REACT_APP_CREATE_JOB_URL=$CREATE_JOB_URL
REACT_APP_PROCESS_JOB_URL=$PROCESS_JOB_URL
REACT_APP_GET_STATUS_URL=$GET_STATUS_URL
REACT_APP_UPLOAD_URL=$UPLOAD_URL
REACT_APP_LIST_JOBS_URL=$LIST_JOBS_URL
EOF
    
    log_info "API Endpoints:"
    log_info "  Health Check: $HEALTH_URL"
    log_info "  Create Job:   $CREATE_JOB_URL"
    log_info "  Process Job:  $PROCESS_JOB_URL"
    log_info "  Get Status:   $GET_STATUS_URL"
    log_info "  Upload URL:   $UPLOAD_URL"
    log_info "  List Jobs:    $LIST_JOBS_URL"
    
    cd "$PROJECT_ROOT"
else
    log_warning "Skipping Terraform deployment"
fi

# Deploy frontend
if [[ "$SKIP_FRONTEND" == false ]]; then
    log_info "Deploying frontend..."
    cd "$PROJECT_ROOT/frontend"
    
    # Install dependencies
    npm install
    
    # Build
    npm run build
    
    # Initialize Firebase if needed
    if [[ ! -f "firebase.json" ]]; then
        log_info "Initializing Firebase..."
        cat > firebase.json << EOF
{
  "hosting": {
    "public": "build",
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
    "rewrites": [{"source": "**", "destination": "/index.html"}]
  }
}
EOF
    fi
    
    # Deploy to Firebase Hosting
    firebase deploy --only hosting --project "$PROJECT_ID"
    
    log_success "Frontend deployed!"
    cd "$PROJECT_ROOT"
else
    log_warning "Skipping frontend deployment"
fi

# Verify deployment
log_info "Verifying deployment..."

if [[ -n "$HEALTH_URL" ]]; then
    HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL")
    if [[ "$HEALTH_STATUS" == "200" ]]; then
        log_success "Health check passed!"
    else
        log_warning "Health check returned status $HEALTH_STATUS"
    fi
fi

log_info "=============================================="
log_success "Deployment Complete!"
log_info "=============================================="
log_info ""
log_info "Next Steps:"
log_info "1. Configure Firebase Authentication in the Firebase Console"
log_info "2. Update frontend/.env.local with your Firebase config"
log_info "3. Rebuild and redeploy frontend: cd frontend && npm run build && firebase deploy"
log_info ""
log_info "Documentation: $PROJECT_ROOT/docs/README.md"
