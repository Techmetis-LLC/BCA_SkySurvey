#!/usr/bin/env python3
"""
Automated Plate Solving Script
Adds WCS to astronomical images using astrometry.net
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from astropy.io import fits
from astropy.wcs import WCS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AstrometrySolver:
    """Automated plate solving using astrometry.net"""
    
    def __init__(self, 
                 scale_low: float = 0.5,
                 scale_high: float = 3.0,
                 scale_units: str = 'arcsecperpix',
                 timeout: int = 300,
                 debug: bool = False):
        """
        Initialize the plate solver.
        
        Parameters:
        -----------
        scale_low : float
            Lower bound of image scale
        scale_high : float
            Upper bound of image scale
        scale_units : str
            Units for scale (arcsecperpix, arcminwidth, degwidth)
        timeout : int
            Timeout in seconds for each solve
        debug : bool
            Enable debug output
        """
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.scale_units = scale_units
        self.timeout = timeout
        self.debug = debug
        
        # Check if solve-field is available
        if not self._check_astrometry_installed():
            raise RuntimeError("astrometry.net not found. Please install it first.")
    
    def _check_astrometry_installed(self) -> bool:
        """Check if astrometry.net is installed."""
        result = shutil.which('solve-field')
        if result:
            logger.info(f"Found astrometry.net at: {result}")
            return True
        else:
            logger.error("astrometry.net (solve-field) not found in PATH")
            return False
    
    def check_wcs(self, fits_path: Path) -> Tuple[bool, Optional[WCS]]:
        """
        Check if a FITS file already has valid WCS.
        
        Returns:
        --------
        Tuple[bool, Optional[WCS]]
            (has_valid_wcs, wcs_object)
        """
        try:
            with fits.open(fits_path) as hdul:
                wcs = WCS(hdul[0].header)
                
                # Check if WCS has celestial coordinates
                if not wcs.has_celestial:
                    return False, None
                
                # Check coordinate types
                ctype = wcs.wcs.ctype
                if len(ctype) < 2:
                    return False, None
                
                # Verify it's RA/Dec
                is_celestial = (
                    ('RA' in ctype[0] or 'GLON' in ctype[0]) and
                    ('DEC' in ctype[1] or 'GLAT' in ctype[1])
                )
                
                if not is_celestial:
                    return False, None
                
                # Test conversion
                try:
                    world = wcs.all_pix2world([[500, 500]], 0)
                    ra, dec = float(world[0][0]), float(world[0][1])
                    
                    if not (0 <= ra <= 360 and -90 <= dec <= 90):
                        return False, None
                    
                    return True, wcs
                except:
                    return False, None
                    
        except Exception as e:
            if self.debug:
                logger.debug(f"WCS check failed for {fits_path}: {e}")
            return False, None
    
    def solve_field(self, 
                   image_path: Path,
                   ra_hint: Optional[float] = None,
                   dec_hint: Optional[float] = None,
                   radius_hint: Optional[float] = None,
                   overwrite: bool = True) -> bool:
        """
        Solve astrometry for a single image.
        
        Parameters:
        -----------
        image_path : Path
            Path to image file
        ra_hint : float, optional
            RA hint in degrees
        dec_hint : float, optional
            Dec hint in degrees
        radius_hint : float, optional
            Search radius in degrees
        overwrite : bool
            Overwrite original file with solved version
        
        Returns:
        --------
        bool : True if solving succeeded
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return False
        
        # Check if already has valid WCS
        if not overwrite:
            has_wcs, _ = self.check_wcs(image_path)
            if has_wcs:
                logger.info(f"✓ {image_path.name} already has valid WCS")
                return True
        
        # Build command
        cmd = [
            'solve-field',
            str(image_path),
            '--no-plots',
            '--no-verify',
            '--scale-units', self.scale_units,
            '--scale-low', str(self.scale_low),
            '--scale-high', str(self.scale_high),
        ]
        
        if overwrite:
            cmd.append('--overwrite')
        
        if ra_hint is not None and dec_hint is not None:
            cmd.extend(['--ra', str(ra_hint)])
            cmd.extend(['--dec', str(dec_hint)])
            if radius_hint:
                cmd.extend(['--radius', str(radius_hint)])
        
        if self.debug:
            cmd.append('-v')
        
        try:
            logger.info(f"Solving {image_path.name}...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                # Check if .new file was created
                solved_file = image_path.parent / f"{image_path.stem}.new"
                
                if solved_file.exists():
                    # Verify WCS in solved file
                    has_wcs, wcs = self.check_wcs(solved_file)
                    
                    if has_wcs:
                        if overwrite:
                            # Replace original with solved version
                            shutil.move(str(solved_file), str(image_path))
                            
                            # Clean up temporary files
                            self._cleanup_temp_files(image_path)
                        
                        logger.info(f"✓ Successfully solved {image_path.name}")
                        
                        # Log field center
                        if wcs:
                            ra_center = wcs.wcs.crval[0]
                            dec_center = wcs.wcs.crval[1]
                            logger.info(f"  Field center: RA={ra_center:.4f}°, Dec={dec_center:.4f}°")
                        
                        return True
                    else:
                        logger.error(f"✗ Solved file has invalid WCS: {image_path.name}")
                        return False
                else:
                    logger.error(f"✗ No .new file created for {image_path.name}")
                    if self.debug:
                        logger.debug(f"stdout: {result.stdout}")
                        logger.debug(f"stderr: {result.stderr}")
                    return False
            else:
                logger.error(f"✗ Solving failed for {image_path.name}")
                if self.debug:
                    logger.debug(f"Return code: {result.returncode}")
                    logger.debug(f"stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout solving {image_path.name} (>{self.timeout}s)")
            return False
        except Exception as e:
            logger.error(f"✗ Error solving {image_path.name}: {e}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            return False
    
    def _cleanup_temp_files(self, image_path: Path):
        """Clean up temporary files created by astrometry.net"""
        temp_extensions = [
            '.axy', '.corr', '.match', '.rdls', '.solved', 
            '.wcs', '-indx.png', '-indx.xyls', '-ngc.png', 
            '-objs.png', '.new'
        ]
        
        for ext in temp_extensions:
            temp_file = image_path.parent / f"{image_path.stem}{ext}"
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
    
    def solve_batch(self, 
                   image_paths: List[Path],
                   parallel: int = 1,
                   skip_existing: bool = True) -> Tuple[int, int]:
        """
        Solve multiple images.
        
        Parameters:
        -----------
        image_paths : List[Path]
            List of image paths
        parallel : int
            Number of parallel solves (1 = sequential)
        skip_existing : bool
            Skip images that already have valid WCS
        
        Returns:
        --------
        Tuple[int, int] : (successful, failed) counts
        """
        successful = 0
        failed = 0
        
        # Filter images if skipping existing
        if skip_existing:
            images_to_solve = []
            for img in image_paths:
                has_wcs, _ = self.check_wcs(img)
                if has_wcs:
                    logger.info(f"⊙ Skipping {img.name} (already has WCS)")
                    successful += 1
                else:
                    images_to_solve.append(img)
        else:
            images_to_solve = image_paths
        
        if not images_to_solve:
            logger.info("All images already have WCS!")
            return successful, failed
        
        logger.info(f"Solving {len(images_to_solve)} images...")
        
        if parallel > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(self.solve_field, img): img 
                    for img in images_to_solve
                }
                
                with tqdm(total=len(futures), desc="Plate solving") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            successful += 1
                        else:
                            failed += 1
                        pbar.update(1)
        else:
            # Sequential processing
            for img in tqdm(images_to_solve, desc="Plate solving"):
                if self.solve_field(img):
                    successful += 1
                else:
                    failed += 1
        
        return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Automated plate solving for astronomical images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve all FITS files in current directory
  %(prog)s *.fits
  
  # Solve with custom scale for wide-field imaging
  %(prog)s --scale-low 5 --scale-high 15 --scale-units arcminwidth *.fits
  
  # Parallel solving with 4 processes
  %(prog)s --parallel 4 *.fits
  
  # Solve with position hint
  %(prog)s --ra 123.45 --dec 45.67 --radius 10 image.fits
        """
    )
    
    parser.add_argument('images', nargs='+', help='Input image files')
    parser.add_argument('--scale-low', type=float, default=0.5,
                       help='Lower bound of image scale (default: 0.5)')
    parser.add_argument('--scale-high', type=float, default=3.0,
                       help='Upper bound of image scale (default: 3.0)')
    parser.add_argument('--scale-units', choices=['arcsecperpix', 'arcminwidth', 'degwidth'],
                       default='arcsecperpix',
                       help='Units for scale (default: arcsecperpix)')
    parser.add_argument('--ra', type=float, help='RA hint in degrees')
    parser.add_argument('--dec', type=float, help='Dec hint in degrees')
    parser.add_argument('--radius', type=float, default=10.0,
                       help='Search radius in degrees (default: 10)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per image in seconds (default: 300)')
    parser.add_argument('--parallel', '-j', type=int, default=1,
                       help='Number of parallel solves (default: 1)')
    parser.add_argument('--force', action='store_true',
                       help='Re-solve images that already have WCS')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose or args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert to Path objects
    image_paths = [Path(p) for p in args.images]
    
    # Validate files exist
    for path in image_paths:
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)
    
    try:
        # Create solver
        solver = AstrometrySolver(
            scale_low=args.scale_low,
            scale_high=args.scale_high,
            scale_units=args.scale_units,
            timeout=args.timeout,
            debug=args.debug
        )
        
        # Solve images
        successful, failed = solver.solve_batch(
            image_paths,
            parallel=args.parallel,
            skip_existing=not args.force
        )
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Plate Solving Summary")
        print(f"{'='*50}")
        print(f"Total images:     {len(image_paths)}")
        print(f"Successfully solved: {successful}")
        print(f"Failed:           {failed}")
        print(f"{'='*50}")
        
        if failed > 0:
            print("\nSome images failed to solve. Common reasons:")
            print("  • Image scale doesn't match specified range")
            print("  • Not enough stars detected")
            print("  • Missing index files for your field of view")
            print("  • Image quality too poor")
            print("\nTry:")
            print("  • Adjusting --scale-low and --scale-high")
            print("  • Downloading additional index files")
            print("  • Using --ra/--dec hints if you know the field")
            sys.exit(1)
        else:
            print("\n✓ All images solved successfully!")
            print("You can now run: python asteroid_detector.py *.fits")
        
    except Exception as e:
        logger.error(f"Plate solving failed: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
