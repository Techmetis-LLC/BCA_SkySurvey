#!/bin/bash
# ==============================================================================
# Asteroid Detection Platform - Local Development Setup
# ==============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "Asteroid Detection Platform - Setup"
echo "=============================================="

# Check Python version
log_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    log_warning "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
log_success "Python $PYTHON_VERSION found"

# Create virtual environment
log_info "Creating Python virtual environment..."
cd "$PROJECT_ROOT"

if [[ -d "venv" ]]; then
    log_warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    log_success "Virtual environment created"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
log_info "Installing Python dependencies..."
pip install -r requirements.txt --quiet

# Verify key imports
log_info "Verifying installation..."
python3 -c "
import numpy
import astropy
import sep
import photutils
import sklearn
print('All key packages imported successfully!')
"
log_success "Installation verified"

# Install frontend dependencies (optional)
if command -v npm &> /dev/null; then
    log_info "Installing frontend dependencies..."
    cd "$PROJECT_ROOT/frontend"
    npm install --quiet
    cd "$PROJECT_ROOT"
    log_success "Frontend dependencies installed"
else
    log_warning "npm not found. Skipping frontend setup."
    log_warning "Install Node.js if you want to use the web interface."
fi

# Create example configuration
log_info "Creating example configuration..."
if [[ ! -f "$PROJECT_ROOT/terraform/terraform.tfvars" ]]; then
    cp "$PROJECT_ROOT/terraform/terraform.tfvars.example" "$PROJECT_ROOT/terraform/terraform.tfvars" 2>/dev/null || true
fi

echo ""
echo "=============================================="
log_success "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run asteroid detection:"
echo "  python src/asteroid_detector.py image1.fits image2.fits image3.fits"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""
echo "For GCP deployment, update terraform/terraform.tfvars with your project ID"
echo "then run: ./scripts/deploy.sh -p YOUR_PROJECT_ID"
echo ""
