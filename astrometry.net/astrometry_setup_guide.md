# Astrometry.net Setup Guide

Complete guide to installing and using astrometry.net for plate solving astronomical images.

## Table of Contents

1. [Installation](#installation)
2. [Index Files Setup](#index-files-setup)
3. [Testing the Installation](#testing-the-installation)
4. [Integration with Detection System](#integration-with-detection-system)
5. [Troubleshooting](#troubleshooting)

---

## Installation

### Linux (Ubuntu/Debian)

```bash
# Install astrometry.net
sudo apt-get update
sudo apt-get install astrometry.net astrometry-data-tycho2

# Install Python bindings
pip install astroquery
```

### macOS

```bash
# Using Homebrew
brew install astrometry-net

# Install Python bindings
pip install astroquery
```

### Windows (WSL Recommended)

```bash
# Install WSL first, then use Ubuntu instructions above
# Or use Docker (see Docker section below)
```

### Using Conda (Cross-platform)

```bash
# Create conda environment
conda create -n astronomy python=3.9
conda activate astronomy

# Install astrometry
conda install -c conda-forge astrometry

# Install required packages
pip install astropy astroquery numpy
```

### Using Docker (Easiest for Windows)

```bash
# Pull astrometry.net Docker image
docker pull astronomerio/astrometry

# Run plate solving
docker run -v /path/to/images:/images astronomerio/astrometry \
    solve-field /images/your_image.fits --overwrite
```

---

## Index Files Setup

Astrometry.net needs index files to match star patterns. Choose based on your telescope's field of view.

### Understanding Index Files

| Index Series | Field of View | Typical Use |
|--------------|---------------|-------------|
| 4200 series | 30° - 2000° | All-sky cameras |
| 4100 series | 10° - 30° | Wide field lenses |
| 5200 series | 2° - 30° | Camera lenses |
| 5000 series | 8' - 2° | Small telescopes |
| 4000 series | 4' - 30' | Medium telescopes |
| Tycho-2 | Down to 4' | Larger telescopes |

### Download Index Files

**Option 1: Download Manually**

```bash
# Create index directory
sudo mkdir -p /usr/share/astrometry
cd /usr/share/astrometry

# Download index files (example for 0.5° - 2° field of view)
# Choose files based on your telescope
sudo wget http://data.astrometry.net/5200/index-5206-00.fits
sudo wget http://data.astrometry.net/5200/index-5206-01.fits
sudo wget http://data.astrometry.net/5200/index-5206-02.fits
# ... download more as needed

# For wider fields (2° - 10°):
sudo wget http://data.astrometry.net/5100/index-5103-00.fits
sudo wget http://data.astrometry.net/5100/index-5103-01.fits

# For narrower fields (10' - 30'):
sudo wget http://data.astrometry.net/4100/index-4115.fits
sudo wget http://data.astrometry.net/4100/index-4116.fits
```

**Option 2: Use Install Script**

```bash
# Download and run installer
wget http://data.astrometry.net/debian/astrometry-data-installer.sh
chmod +x astrometry-data-installer.sh

# Install for your field of view (in arcminutes)
# For 30 arcminute FOV:
./astrometry-data-installer.sh 30
```

**Option 3: Install Package (Limited)**

```bash
# Ubuntu/Debian - installs basic Tycho-2 data
sudo apt-get install astrometry-data-tycho2
```

### Recommended Index Files by Telescope

**DSLR with 50mm lens (~10° FOV):**
```bash
cd /usr/share/astrometry
sudo wget http://data.astrometry.net/5100/index-5103-{00..11}.fits
```

**Small telescope 200mm focal length (~2° FOV):**
```bash
cd /usr/share/astrometry
sudo wget http://data.astrometry.net/5200/index-5206-{00..11}.fits
```

**Medium telescope 1000mm focal length (~30' FOV):**
```bash
cd /usr/share/astrometry
sudo wget http://data.astrometry.net/4100/index-411{2..8}.fits
```

### Configure Index Path

Edit the configuration file:

```bash
sudo nano /etc/astrometry.cfg
```

Add/verify these lines:
```
# Add index file directory
add_path /usr/share/astrometry

# Or if you put them elsewhere:
add_path /path/to/your/index/files
```

---

## Testing the Installation

### Test 1: Command Line Plate Solving

```bash
# Basic solve
solve-field your_image.fits

# With options (recommended)
solve-field your_image.fits \
    --scale-units arcsecperpix \
    --scale-low 0.5 \
    --scale-high 2.0 \
    --overwrite \
    --no-plots

# Expected output:
# Reading input file...
# Solving...
# Field center: (RA,Dec) = (123.456, +45.678)
# Field size: 1.5 x 1.0 degrees
# Success!
```

### Test 2: Python API

Create `test_astrometry.py`:

```python
from astropy.io import fits
from astropy.wcs import WCS
import subprocess
import sys

def test_solve(image_path):
    print(f"Testing plate solve on: {image_path}")
    
    # Run solve-field
    cmd = [
        'solve-field',
        image_path,
        '--overwrite',
        '--no-plots',
        '--no-verify',
        '--scale-units', 'arcsecperpix',
        '--scale-low', '0.5',
        '--scale-high', '3.0'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Plate solving successful!")
            
            # Check WCS
            solved_file = image_path.replace('.fits', '.new')
            with fits.open(solved_file) as hdul:
                wcs = WCS(hdul[0].header)
                print(f"✓ WCS found: {wcs.wcs.ctype}")
                
                # Test conversion
                ra, dec = wcs.all_pix2world([[500, 500]], 0)[0]
                print(f"✓ Test pixel (500,500) -> RA={ra:.4f}°, Dec={dec:.4f}°")
                
            return True
        else:
            print("✗ Plate solving failed")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout - solving took too long")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_astrometry.py image.fits")
        sys.exit(1)
    
    test_solve(sys.argv[1])
```

Run the test:
```bash
python test_astrometry.py your_image.fits
```

### Test 3: Verify WCS Headers

```python
from astropy.io import fits
from astropy.wcs import WCS

# Check original file
with fits.open('original.fits') as hdul:
    print("Original WCS:")
    try:
        wcs = WCS(hdul[0].header)
        print(f"  Has celestial: {wcs.has_celestial}")
        print(f"  CTYPE: {wcs.wcs.ctype}")
    except:
        print("  No valid WCS")

# Check solved file
with fits.open('original.new') as hdul:
    print("\nSolved WCS:")
    wcs = WCS(hdul[0].header)
    print(f"  Has celestial: {wcs.has_celestial}")
    print(f"  CTYPE: {wcs.wcs.ctype}")
    print(f"  CRVAL: {wcs.wcs.crval}")
    print(f"  Field center: RA={wcs.wcs.crval[0]:.4f}°, Dec={wcs.wcs.crval[1]:.4f}°")
```

---

## Integration with Detection System

I've created three integration options:

### Option 1: Pre-Process Images (Simplest)

```bash
# Create a script: prepare_images.sh
#!/bin/bash

echo "Plate-solving all FITS images..."

for img in *.fits; do
    echo "Processing $img..."
    solve-field "$img" \
        --overwrite \
        --no-plots \
        --scale-units arcsecperpix \
        --scale-low 0.5 \
        --scale-high 3.0
    
    # Rename solved file back to original
    if [ -f "${img%.fits}.new" ]; then
        mv "${img%.fits}.new" "$img"
        echo "✓ $img solved successfully"
    else
        echo "✗ Failed to solve $img"
    fi
done

echo "Done! Now run: python asteroid_detector.py *.fits"
```

Make executable and run:
```bash
chmod +x prepare_images.sh
./prepare_images.sh
python asteroid_detector.py *.fits
```

### Option 2: Automatic Solving in Python

I'll create an enhanced version of the detector that automatically solves images.

### Option 3: Online Astrometry.net API

For occasional use or if local installation isn't working.

---

## Troubleshooting

### Issue: "solve-field: command not found"

**Solution:**
```bash
# Check installation
which solve-field

# If not found, reinstall
sudo apt-get install --reinstall astrometry.net

# Or add to PATH
export PATH=$PATH:/usr/local/astrometry/bin
```

### Issue: "No index files found"

**Solution:**
```bash
# Check index directory
ls -lh /usr/share/astrometry/

# Download missing indexes
cd /usr/share/astrometry
sudo wget http://data.astrometry.net/5200/index-5206-00.fits

# Update config
sudo nano /etc/astrometry.cfg
# Add: add_path /usr/share/astrometry
```

### Issue: Solving is very slow

**Solutions:**
1. **Reduce search area:**
   ```bash
   solve-field image.fits \
       --scale-low 0.8 \
       --scale-high 1.2  # Narrow range
   ```

2. **Provide hint coordinates:**
   ```bash
   solve-field image.fits \
       --ra 123.45 \
       --dec 45.67 \
       --radius 5  # Search within 5 degrees
   ```

3. **Limit depth:**
   ```bash
   solve-field image.fits \
       --depth 10,20,30,40,50  # Don't go too deep
   ```

### Issue: Solving fails for all images

**Diagnostic steps:**
```bash
# 1. Check image has stars
solve-field image.fits --plot-all

# 2. Try with verbose output
solve-field image.fits -v

# 3. Check if index files match your FOV
solve-field image.fits --scale-units degwidth --scale-low 0.5 --scale-high 3.0

# 4. Test with a known-good image from online
wget http://data.astrometry.net/demo/demo1.jpg
solve-field demo1.jpg
```

### Issue: "Failed to get WCS header"

**Solution:**
The solved file has a different name. Check for:
- `image.new` (main solved file)
- `image.solved` (flag file)
- `image.wcs` (WCS-only file)

```bash
# Use the .new file
cp image.new image.fits
```

---

## Quick Reference

### Common Commands

```bash
# Basic solve
solve-field image.fits --overwrite

# Fast solve (skip verification)
solve-field image.fits --overwrite --no-verify --no-plots

# Solve with scale hint (1 arcsec/pixel ± 20%)
solve-field image.fits --scale-units arcsecperpix --scale-low 0.8 --scale-high 1.2

# Solve with position hint
solve-field image.fits --ra 123.45 --dec 45.67 --radius 10

# Batch processing
for f in *.fits; do solve-field "$f" --overwrite --no-plots; done

# Use multicore
solve-field image.fits --overwrite --cpulimit 300 -j 4
```

### Recommended Settings by Telescope

**DSLR + 50mm lens:**
```bash
solve-field image.fits \
    --scale-units arcminwidth \
    --scale-low 300 \
    --scale-high 900 \
    --overwrite
```

**Small telescope (500mm):**
```bash
solve-field image.fits \
    --scale-units arcsecperpix \
    --scale-low 1.0 \
    --scale-high 3.0 \
    --overwrite
```

**Medium telescope (2000mm):**
```bash
solve-field image.fits \
    --scale-units arcsecperpix \
    --scale-low 0.3 \
    --scale-high 1.0 \
    --overwrite
```

---

## Next Steps

1. **Install astrometry.net** using the method for your OS
2. **Download index files** matching your telescope's field of view
3. **Test with one image** using `solve-field`
4. **Batch process** all your images with the prepare_images.sh script
5. **Run the detector** with plate-solved images

Once your images have proper WCS, the detection system will automatically:
- ✅ Extract sky coordinates for all detections
- ✅ Query SkyBoT for object identification
- ✅ Generate reports with RA/Dec coordinates
- ✅ Identify known asteroids and comets

Need help with any of these steps? Let me know!