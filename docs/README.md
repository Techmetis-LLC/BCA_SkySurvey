# Asteroid Detection Platform

A comprehensive system for detecting moving objects (asteroids, comets) in astronomical image sequences. The platform identifies objects that move differently from the stellar background, performs plate solving for celestial coordinates, and queries NASA/Minor Planet Center databases to identify known objects or flag potential new discoveries.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Local Installation](#local-installation)
- [GCP Deployment](#gcp-deployment)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Features

### Core Detection Capabilities

- **Multi-format Support**: FITS, TIFF, JPEG, and XISF image formats
- **Automated Source Detection**: Using SEP (Source Extractor Python) for stellar object identification
- **Image Registration**: Automatic alignment to compensate for telescope tracking errors
- **Motion Detection**: Identifies objects moving against the stellar background
- **Plate Solving**: Determines precise celestial coordinates (via existing WCS or astrometry.net)
- **Database Integration**: Queries NASA JPL Horizons and Minor Planet Center databases
- **Discovery Classification**: Distinguishes known objects from potential new discoveries

### Technical Features

- **Progress Tracking**: Real-time progress indication with multiple output modes
- **Debug Mode**: Comprehensive logging for troubleshooting
- **Parallel Processing**: Multi-threaded image loading and processing
- **Cloud-Native**: Deployable on Google Cloud Platform with serverless architecture
- **REST API**: Full-featured API for integration with other systems

## System Architecture

```
┌───────────────────────────────────────────────────────────────────────-──┐
│                        Asteroid Detection Platform                       │
├──────────────────────────────────────────────────────────────────────-───┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐  │
│  │   Frontend   │────▶│ Cloud Funcs  │────▶│     Processing Core      │  │
│  │   (React)    │     │  (Python)    │     │                          │  │
│  └──────────────┘     └──────────────┘     │  ┌────────────────────┐  │  │
│         │                    │             │  │  ImageProcessor    │  │  │
│         │                    │             │  ├────────────────────┤  │  │
│         ▼                    ▼             │  │  StarDetector      │  │  │
│  ┌──────────────┐     ┌──────────────┐     │  ├────────────────────┤  │  │
│  │   Firebase   │     │ Cloud Storage│     │  │  ImageRegistrar    │  │  │
│  │     Auth     │     │   (Images)   │     │  ├────────────────────┤  │  │
│  └──────────────┘     └──────────────┘     │  │  MotionDetector    │  │  │
│                              │             │  ├────────────────────┤  │  │
│                              ▼             │  │  PlateSolver       │  │  │
│                       ┌──────────────┐     │  ├────────────────────┤  │  │
│                       │  Firestore   │     │  │  ObjectIdentifier  │  │  │
│                       │   (Jobs)     │     │  └────────────────────┘  │  │
│                       └──────────────┘     └──────────────────────────┘  │
│                                                         │                │
│                                            ┌────────────┴────────────┐   │
│                                            ▼                         ▼   │
│                                     ┌─────────────┐          ┌──────────┐│
│                                     │ JPL Horizons│          │   MPC    ││
│                                     └─────────────┘          └──────────┘│
└────────────────────────────────────────────────────────────────────────-─┘
```

## Quick Start

### Option 1: Local Standalone (Fastest)

```bash
# Clone or download the project
cd BCA_SkySurvey

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run detection on your images
python src/asteroid_detector.py image1.fits image2.fits image3.fits
```

### Option 2: Docker (Recommended for consistency)

```bash
# Build the Docker image
docker build -t BCA_SkySurvey .

# Run detection
docker run -v $(pwd)/images:/data BCA_SkySurvey \
    python src/asteroid_detector.py /data/*.fits
```

### Option 3: GCP Cloud Deployment

```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"

# Run deployment script
./scripts/deploy.sh -p $PROJECT_ID

# Access web interface at:
# https://your-project-id.web.app
```

## Local Installation

### Prerequisites

- Python 3.9 or later
- pip (Python package manager)
- Git (optional, for cloning)

### Step-by-Step Installation

1. **Create project directory and virtual environment:**

```bash
mkdir asteroid-detection && cd asteroid-detection
python3 -m venv venv
source venv/bin/activate
```

2. **Install required packages:**

```bash
pip install --upgrade pip
pip install numpy scipy matplotlib
pip install astropy astroquery photutils sep
pip install scikit-image scikit-learn opencv-python Pillow
pip install tqdm requests
```

3. **Verify installation:**

```bash
python -c "import sep; import astropy; print('Installation successful!')"
```

### System Dependencies (Linux/macOS)

For optimal performance, install these system packages:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libfftw3-dev libatlas-base-dev

# macOS (with Homebrew)
brew install fftw
```

## GCP Deployment

### Prerequisites

- Google Cloud Platform account with billing enabled
- Google Cloud SDK (`gcloud`) installed and configured
- Terraform >= 1.0 installed
- Node.js and npm (for frontend)
- Firebase CLI (`npm install -g firebase-tools`)

### Deployment Steps

#### 1. Set Up GCP Project

```bash
# Set project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
    cloudfunctions.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    firestore.googleapis.com \
    firebase.googleapis.com \
    run.googleapis.com
```

#### 2. Configure Firebase

```bash
# Login to Firebase
firebase login

# Initialize Firebase in your project
firebase init

# Select:
# - Firestore
# - Hosting
# - Authentication (Google provider)
```

#### 3. Deploy Infrastructure

```bash
# Navigate to terraform directory
cd terraform

# Create your configuration
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID

# Initialize and deploy
terraform init
terraform apply
```

#### 4. Deploy Frontend

```bash
cd frontend
npm install
npm run build
firebase deploy --only hosting
```

#### 5. Configure Authentication

1. Go to Firebase Console → Authentication → Sign-in method
2. Enable Google as a sign-in provider
3. Add your domain to authorized domains
4. Copy Firebase config to `frontend/.env.local`

### Automated Deployment

Use the provided deployment script for one-command deployment:

```bash
./scripts/deploy.sh -p your-project-id -e dev
```

Options:
- `-p PROJECT_ID`: GCP Project ID (required)
- `-r REGION`: GCP region (default: us-central1)
- `-e ENVIRONMENT`: Environment name (default: dev)
- `--skip-terraform`: Skip infrastructure deployment
- `--skip-frontend`: Skip frontend deployment

## Usage Guide

### Command Line Interface

```bash
# Basic usage
python src/asteroid_detector.py image1.fits image2.fits image3.fits

# With options
python src/asteroid_detector.py \
    --threshold 3.0 \
    --min-detections 3 \
    --output results.json \
    --format json \
    --verbose \
    *.fits

# Debug mode with log file
python src/asteroid_detector.py \
    --debug \
    --log-file detection.log \
    images/*.fits
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output file path | stdout |
| `--format` | `-f` | Output format (json, markdown, text) | text |
| `--threshold` | `-t` | Detection threshold in sigma | 3.0 |
| `--min-detections` | `-m` | Minimum detections for confirmation | 3 |
| `--astrometry-key` | | API key for astrometry.net | None |
| `--verbose` | `-v` | Enable verbose output | False |
| `--debug` | | Enable debug mode | False |
| `--no-progress` | | Disable progress bar | False |
| `--log-file` | | Write logs to file | None |

### Input Image Requirements

- **Minimum Images**: At least 2 images required for motion detection
- **Formats**: FITS (.fits, .fit, .fts), TIFF (.tiff, .tif), JPEG (.jpg, .jpeg), XISF (.xisf)
- **Time Separation**: Images should be taken at least 5-10 minutes apart
- **Field of View**: All images should cover the same sky region
- **WCS Header**: FITS files with WCS headers provide better coordinate accuracy

### Web Interface Usage

1. **Sign In**: Click "Sign In with Google" to authenticate
2. **Upload Images**: Drag and drop or click to select astronomical images
3. **Configure Options**: Adjust detection threshold and minimum detections
4. **Start Detection**: Click "Start Detection" to begin processing
5. **View Results**: Review detected objects, download reports

## API Reference

### REST API Endpoints

All endpoints require Firebase authentication (Bearer token in Authorization header).

#### Create Detection Job

```http
POST /asteroid-create-job-{env}
Content-Type: application/json
Authorization: Bearer {firebase_token}

{
    "image_urls": [
        "gs://bucket/image1.fits",
        "gs://bucket/image2.fits"
    ],
    "options": {
        "threshold": 3.0,
        "min_detections": 3
    }
}
```

Response:
```json
{
    "job_id": "uuid",
    "status": "pending",
    "message": "Detection job created successfully"
}
```

#### Get Job Status

```http
GET /asteroid-get-status-{env}?job_id={job_id}
Authorization: Bearer {firebase_token}
```

Response:
```json
{
    "job_id": "uuid",
    "status": "completed",
    "progress": 100,
    "result": {
        "moving_objects_count": 3,
        "known_objects_count": 2,
        "potential_discoveries_count": 1,
        "moving_objects": [...]
    }
}
```

#### Get Signed Upload URL

```http
POST /asteroid-upload-url-{env}
Content-Type: application/json
Authorization: Bearer {firebase_token}

{
    "filename": "image.fits",
    "content_type": "application/fits"
}
```

Response:
```json
{
    "upload_url": "https://storage.googleapis.com/...",
    "blob_url": "gs://bucket/uploads/user/uuid/image.fits",
    "expires_in": 3600
}
```

#### List User Jobs

```http
GET /asteroid-list-jobs-{env}?limit=20&status=completed
Authorization: Bearer {firebase_token}
```

### Python API

```python
from asteroid_detector import AsteroidDetector
from pathlib import Path

# Initialize detector
detector = AsteroidDetector(
    debug=False,
    verbose=True,
    progress_mode="tqdm",
    astrometry_api_key="your-api-key"  # Optional
)

# Run detection
result = detector.detect(
    image_paths=[Path("img1.fits"), Path("img2.fits"), Path("img3.fits")],
    detection_threshold=3.0,
    min_detections=3
)

# Access results
print(f"Found {len(result.moving_objects)} moving objects")
print(f"Known objects: {len(result.known_objects)}")
print(f"Potential discoveries: {len(result.potential_discoveries)}")

# Generate report
report = detector.generate_report(result, Path("report.md"))
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STORAGE_BUCKET` | GCS bucket for uploads | - |
| `RESULTS_BUCKET` | GCS bucket for results | - |
| `ENVIRONMENT` | Environment name | dev |

### Frontend Configuration

Create `frontend/.env.local`:

```bash
REACT_APP_FIREBASE_API_KEY=your-api-key
REACT_APP_FIREBASE_AUTH_DOMAIN=project.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=project-id
REACT_APP_FIREBASE_STORAGE_BUCKET=project.appspot.com
REACT_APP_FIREBASE_MESSAGING_SENDER_ID=123456789
REACT_APP_FIREBASE_APP_ID=1:123:web:abc
REACT_APP_API_URL=https://region-project.cloudfunctions.net
```

### Detection Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `threshold` | Detection significance (σ above background) | 1.0-10.0 | 3.0 |
| `min_detections` | Minimum frames for confirmed detection | 2-10 | 3 |
| `min_area` | Minimum source area in pixels | 1-20 | 5 |

## Troubleshooting

### Common Issues

#### "No module named 'sep'"

```bash
pip install sep
# If that fails, try:
pip install --no-cache-dir sep
```

#### "FITS file has no image data"

Ensure your FITS file contains image data in the primary HDU or first extension:
```python
from astropy.io import fits
with fits.open('image.fits') as hdul:
    print(hdul.info())  # Check which HDU has data
```

#### "Motion detection failed: Need at least 2 valid images"

- Ensure all images are readable and contain valid data
- Check that images cover the same sky region
- Verify images have different observation times

#### Cloud Functions timeout

For large images, increase the timeout in `terraform/main.tf`:
```hcl
service_config {
    timeout_seconds = 540  # 9 minutes
}
```

### Debug Mode

Enable comprehensive logging:

```bash
python src/asteroid_detector.py --debug --log-file debug.log images/*.fits
```

Review the log file for detailed processing information.

### Getting Help

1. Check the [Troubleshooting](#troubleshooting) section
2. Review debug logs
3. Open an issue on GitHub with:
   - Python version (`python --version`)
   - Package versions (`pip freeze`)
   - Error message and stack trace
   - Sample images (if possible)

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

### Development Setup

```bash
# Clone repository
git clone https://github.com/Techmetis-LLC/BSA_SkySurvey
cd BCA_SkySurvey

# Create development environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_detector.py -v
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Astropy](https://www.astropy.org/) - Core astronomy library
- [SEP](https://sep.readthedocs.io/) - Source Extractor as a Python library
- [Astroquery](https://astroquery.readthedocs.io/) - Database query tools
- [Google Cloud Platform](https://cloud.google.com/) - Cloud infrastructure

---

**Happy Asteroid Hunting! **
