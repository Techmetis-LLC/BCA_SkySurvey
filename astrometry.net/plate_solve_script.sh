#!/bin/bash
# Quick plate solving script for astronomical images
# Usage: ./plate_solve.sh *.fits

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration - ADJUST THESE FOR YOUR TELESCOPE
SCALE_LOW=0.5          # Lower scale bound (arcsec/pixel)
SCALE_HIGH=3.0         # Upper scale bound (arcsec/pixel)
SCALE_UNITS="arcsecperpix"  # or "arcminwidth" or "degwidth"
TIMEOUT=300            # Timeout per image (seconds)

# Counters
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_skip() {
    echo -e "${YELLOW}[⊙]${NC} $1"
}

# Check if solve-field is installed
if ! command -v solve-field &> /dev/null; then
    echo -e "${RED}ERROR: astrometry.net (solve-field) not found!${NC}"
    echo ""
    echo "Please install astrometry.net first:"
    echo "  Ubuntu/Debian: sudo apt-get install astrometry.net"
    echo "  macOS: brew install astrometry-net"
    echo "  Conda: conda install -c conda-forge astrometry"
    exit 1
fi

# Check if any files provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_files>"
    echo ""
    echo "Examples:"
    echo "  $0 *.fits              # Solve all FITS files"
    echo "  $0 image1.fits image2.fits"
    echo "  $0 /path/to/images/*.fits"
    exit 1
fi

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║     Automated Plate Solving for Astronomy Images     ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
log_info "Configuration:"
echo "  Scale: $SCALE_LOW - $SCALE_HIGH $SCALE_UNITS"
echo "  Timeout: ${TIMEOUT}s per image"
echo "  Images to process: $#"
echo ""

# Function to check if file has valid WCS
check_wcs() {
    local file="$1"
    
    # Use Python to check WCS if available
    if command -v python3 &> /dev/null; then
        python3 << EOF 2>/dev/null
from astropy.io import fits
from astropy.wcs import WCS
import sys

try:
    with fits.open('$file') as hdul:
        wcs = WCS(hdul[0].header)
        if wcs.has_celestial:
            ctype = wcs.wcs.ctype
            if len(ctype) >= 2 and ('RA' in ctype[0] or 'GLON' in ctype[0]):
                # Test conversion
                world = wcs.all_pix2world([[500, 500]], 0)
                ra, dec = float(world[0][0]), float(world[0][1])
                if 0 <= ra <= 360 and -90 <= dec <= 90:
                    sys.exit(0)  # Has valid WCS
except:
    pass
sys.exit(1)  # No valid WCS
EOF
        return $?
    else
        # Fallback: check if .solved file exists
        if [ -f "${file%.fits}.solved" ]; then
            return 0
        fi
        return 1
    fi
}

# Process each image
for image in "$@"; do
    TOTAL=$((TOTAL + 1))
    
    if [ ! -f "$image" ]; then
        log_error "File not found: $image"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Check if already solved
    if check_wcs "$image"; then
        log_skip "$(basename "$image") - already has valid WCS"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    log_info "Solving $(basename "$image")..."
    
    # Run solve-field
    if timeout ${TIMEOUT} solve-field "$image" \
        --overwrite \
        --no-plots \
        --no-verify \
        --scale-units "$SCALE_UNITS" \
        --scale-low "$SCALE_LOW" \
        --scale-high "$SCALE_HIGH" \
        > /tmp/solve_field_$$.log 2>&1; then
        
        # Check if .new file was created
        new_file="${image%.fits}.new"
        if [ -f "$new_file" ]; then
            # Replace original with solved version
            mv "$new_file" "$image"
            
            # Clean up temporary files
            rm -f "${image%.fits}.axy" \
                  "${image%.fits}.corr" \
                  "${image%.fits}.match" \
                  "${image%.fits}.rdls" \
                  "${image%.fits}.solved" \
                  "${image%.fits}.wcs" \
                  "${image%.fits}"-*.png \
                  "${image%.fits}"-*.xyls
            
            log_success "$(basename "$image") solved successfully"
            SUCCESS=$((SUCCESS + 1))
        else
            log_error "$(basename "$image") - no output file created"
            FAILED=$((FAILED + 1))
        fi
    else
        log_error "$(basename "$image") - solving failed or timed out"
        FAILED=$((FAILED + 1))
        
        # Show last few lines of log for debugging
        if [ -f /tmp/solve_field_$$.log ]; then
            echo "  Last error: $(tail -1 /tmp/solve_field_$$.log)"
        fi
    fi
    
    # Clean up log
    rm -f /tmp/solve_field_$$.log
done

# Print summary
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║                    Summary                            ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo "  Total images:        $TOTAL"
echo "  Successfully solved: $SUCCESS"
echo "  Already had WCS:     $SKIPPED"
echo "  Failed:              $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}Some images failed to solve. Common reasons:${NC}"
    echo "  • Image scale doesn't match configured range"
    echo "  • Not enough stars in the image"
    echo "  • Missing index files for your field of view"
    echo "  • Image quality too poor"
    echo ""
    echo "Try adjusting SCALE_LOW and SCALE_HIGH at the top of this script."
    echo "For your telescope, calculate: plate_scale = 206265 * pixel_size_mm / focal_length_mm"
    exit 1
else
    echo -e "${GREEN}✓ All images processed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify WCS: python3 -c \"from astropy.io import fits; from astropy.wcs import WCS; wcs=WCS(fits.open('$1')[0].header); print(wcs)\""
    echo "  2. Run detection: python asteroid_detector.py *.fits"
fi
