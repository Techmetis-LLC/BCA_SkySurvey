# BCA Asteroid Detection Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GCP](https://img.shields.io/badge/GCP-Ready-green.svg)](https://cloud.google.com/)

A robust and scalable system for detecting moving objects (asteroids, comets coming soon) in astronomical image sequences. Identify potential new discoveries or match with known objects in NASA/Minor Planet Center databases.

## Features

- **Multi-format Support**: FITS, TIFF, JPEG, XISF
- **Automated Detection**: SEP-based source extraction with motion analysis
- **Plate Solving**: Celestial coordinate determination
- **Database Integration**: NASA JPL Horizons & Minor Planet Center queries
- **Cloud-Ready**: Deploy to GCP with Terraform
- **Progress Tracking**: Real-time processing status
- **Debug Mode**: Comprehensive logging

## Quick Start

### Local Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/Techmetis-LLC/BSA_SkySurvey
cd BCA_SkySurvey

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Activate environment
source venv/bin/activate

# Run detection
python src/asteroid_detector.py image1.fits image2.fits image3.fits
```

### GCP Cloud Deployment

```bash
# Set your GCP project
export PROJECT_ID="your-project-id"

# Deploy everything
./scripts/deploy.sh -p $PROJECT_ID
```

## Usage

### Command Line

```bash
# Basic usage
python src/asteroid_detector.py *.fits

# With options
python src/asteroid_detector.py \
    --threshold 3.0 \
    --min-detections 3 \
    --output results.json \
    --format json \
    --verbose \
    images/*.fits

# Debug mode
python src/asteroid_detector.py --debug --log-file detection.log images/*.fits
```

### Python API

```python
from asteroid_detector import AsteroidDetector
from pathlib import Path

detector = AsteroidDetector(debug=True, verbose=True)

result = detector.detect(
    image_paths=[Path("img1.fits"), Path("img2.fits"), Path("img3.fits")],
    detection_threshold=3.0,
    min_detections=3
)

print(f"Found {len(result.moving_objects)} moving objects")
print(f"Potential discoveries: {len(result.potential_discoveries)}")
```

## Project Structure

```
asteroid-detection-platform/
├── src/                    # Core detection module
│   └── asteroid_detector.py
├── functions/              # GCP Cloud Functions
├── frontend/               # React web interface
├── terraform/              # Infrastructure as Code
├── scripts/                # Deployment & setup scripts
├── tests/                  # Unit tests
├── docs/                   # Full documentation
└── requirements.txt        # Python dependencies
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output file path | stdout |
| `-f, --format` | Output format (json/markdown/text) | text |
| `-t, --threshold` | Detection threshold (σ) | 3.0 |
| `-m, --min-detections` | Min detections for confirmation | 3 |
| `-v, --verbose` | Enable verbose output | False |
| `--debug` | Enable debug mode | False |
| `--no-progress` | Disable progress bar | False |

## Output Example

```json
{
  "processing_time": 12.5,
  "moving_objects_count": 3,
  "moving_objects": [
    {
      "id": "obj_0001",
      "velocity_arcsec_per_hour": 45.2,
      "position_angle": 127.3,
      "confidence": 0.95,
      "is_known": true,
      "matched_name": "(433) Eros"
    },
    {
      "id": "obj_0002",
      "velocity_arcsec_per_hour": 67.4,
      "position_angle": 256.9,
      "confidence": 0.72,
      "is_known": false,
      "matched_name": null
    }
  ]
}
```

## Documentation

Full documentation is available in [docs/README.md](docs/README.md), including:

- Detailed installation instructions
- GCP deployment guide
- API reference
- Configuration options
- Troubleshooting guide

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Astropy](https://www.astropy.org/) - Core astronomy library
- [SEP](https://sep.readthedocs.io/) - Source Extractor Python
- [Astroquery](https://astroquery.readthedocs.io/) - Database queries

---

**Happy Hunting!**
