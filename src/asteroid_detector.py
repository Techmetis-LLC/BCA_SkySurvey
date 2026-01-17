#!/usr/bin/env python3
"""
Astronomical Object Detection System
Detects moving objects (asteroids, comets) in astronomical image sequences
by identifying objects that don't follow stellar motion patterns.
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
import astropy.visualization as vis
from astroquery.jplhorizons import Horizons
from astroquery.mpc import MPC
import sep
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from skimage import io as skio
from skimage.registration import phase_cross_correlation
from sklearn.cluster import DBSCAN
import cv2
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='astropy.wcs')
warnings.filterwarnings('ignore', message='.*datfix.*')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles loading and preprocessing of astronomical images."""
    
    def __init__(self, debug: bool = False, max_dimension: Optional[int] = None):
        self.debug = debug
        self.max_dimension = max_dimension  # Maximum image dimension for downsampling
        self.supported_formats = ['.jpg', '.jpeg', '.tiff', '.tif', '.fits', '.fit', '.xisf']
    
    def load_image(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load an astronomical image and extract metadata."""
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {suffix}")
        
        metadata = {'filepath': str(filepath), 'format': suffix}
        
        try:
            if suffix in ['.fits', '.fit']:
                data, metadata = self._load_fits(filepath, metadata)
            elif suffix == '.xisf':
                data, metadata = self._load_xisf(filepath, metadata)
            else:
                data, metadata = self._load_standard_image(filepath, metadata)
            
            # Downsample if image is too large
            if self.max_dimension and max(data.shape) > self.max_dimension:
                data, metadata = self._downsample_image(data, metadata)
            
            return data, metadata
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise
    
    def _downsample_image(self, data: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """Downsample large images to save memory."""
        original_shape = data.shape
        
        # Calculate downsampling factor
        max_dim = max(data.shape)
        scale_factor = self.max_dimension / max_dim
        
        new_height = int(data.shape[0] * scale_factor)
        new_width = int(data.shape[1] * scale_factor)
        
        logger.info(f"Downsampling image from {original_shape} to ({new_height}, {new_width}) "
                   f"to reduce memory usage (scale={scale_factor:.2f})")
        
        # Use cv2 for fast downsampling
        downsampled = cv2.resize(data.astype(np.float32), 
                                (new_width, new_height), 
                                interpolation=cv2.INTER_AREA)
        
        metadata['downsampled'] = True
        metadata['original_shape'] = original_shape
        metadata['scale_factor'] = scale_factor
        
        # Update WCS if present
        if 'wcs' in metadata and metadata['wcs'] is not None:
            # Scale the WCS reference pixel
            try:
                wcs = metadata['wcs']
                wcs.wcs.crpix[0] *= scale_factor
                wcs.wcs.crpix[1] *= scale_factor
                # Note: CD matrix should also be scaled but this is approximate
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                    wcs.wcs.cd /= scale_factor
                metadata['wcs'] = wcs
            except Exception as e:
                logger.warning(f"Could not scale WCS: {e}")
                metadata['wcs'] = None
        
        return downsampled.astype(np.float64), metadata
    
    def _load_fits(self, filepath: Path, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """Load FITS image."""
        with fits.open(filepath) as hdul:
            data = hdul[0].data.astype(np.float64)
            header = hdul[0].header
            
            # Extract WCS if present
            try:
                wcs = WCS(header)
                metadata['wcs'] = wcs
            except:
                logger.warning(f"Could not extract WCS from {filepath}")
                metadata['wcs'] = None
            
            # Extract observation time
            for time_key in ['DATE-OBS', 'DATE', 'MJD-OBS']:
                if time_key in header:
                    try:
                        metadata['obs_time'] = Time(header[time_key])
                        break
                    except:
                        continue
            
            metadata['header'] = dict(header)
            
        return data, metadata
    
    def _load_xisf(self, filepath: Path, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """Load XISF image (PixInsight format)."""
        try:
            import xisf
            
            # Open XISF file
            xisf_file = xisf.XISF(str(filepath))
            
            # Read image data
            # The file_meta contains information about images in the file
            if hasattr(xisf_file, 'read_image'):
                data = xisf_file.read_image(0)
            elif hasattr(xisf_file, 'get_images_data'):
                images_data = xisf_file.get_images_data()
                if images_data and len(images_data) > 0:
                    data = images_data[0]
                else:
                    raise ValueError("No image data found in XISF file")
            else:
                raise ValueError("Unsupported xisf library version")
            
            # Handle different array dimensions
            original_shape = data.shape
            
            if self.debug:
                logger.debug(f"XISF raw data shape: {original_shape}, dtype: {data.dtype}")
            
            # Convert multi-dimensional arrays to 2D
            if len(data.shape) == 3:
                # Could be (height, width, channels) or (channels, height, width)
                if data.shape[2] <= 4:
                    # Likely (height, width, channels) - RGB or RGBA
                    logger.info(f"Converting {data.shape[2]}-channel XISF to grayscale")
                    if data.shape[2] == 1:
                        # Single channel - just squeeze
                        data = data[:, :, 0]
                    else:
                        # Multi-channel - average or take first channel
                        # For astronomical images, channels might be RGB or narrowband
                        # Taking the mean is usually safe
                        data = np.mean(data, axis=2)
                elif data.shape[0] <= 4:
                    # Likely (channels, height, width)
                    logger.info(f"Converting {data.shape[0]}-channel XISF to grayscale")
                    if data.shape[0] == 1:
                        data = data[0, :, :]
                    else:
                        data = np.mean(data, axis=0)
                else:
                    # Ambiguous - try to figure out which is the channel dimension
                    # Usually the smallest dimension is channels
                    min_dim = data.shape.index(min(data.shape))
                    logger.info(f"Converting multi-dimensional XISF (shape={data.shape}) to grayscale")
                    if min_dim == 0:
                        data = np.mean(data, axis=0)
                    elif min_dim == 2:
                        data = np.mean(data, axis=2)
                    else:
                        # Middle dimension is smallest - unusual but handle it
                        data = np.mean(data, axis=1)
            
            elif len(data.shape) == 4:
                # Very unusual - might be (batch, channels, height, width) or similar
                logger.warning(f"XISF has 4D data (shape={data.shape}), attempting to reduce to 2D")
                # Take first batch and average channels
                if data.shape[0] == 1:
                    data = data[0]  # Remove batch dimension
                else:
                    data = data[0]  # Take first batch
                
                # Now handle as 3D
                if len(data.shape) == 3:
                    if data.shape[0] <= 4:
                        data = np.mean(data, axis=0)
                    elif data.shape[2] <= 4:
                        data = np.mean(data, axis=2)
            
            elif len(data.shape) > 4:
                raise ValueError(f"XISF data has too many dimensions: {data.shape}")
            
            # Ensure we have a 2D array
            if len(data.shape) != 2:
                raise ValueError(f"Could not convert XISF to 2D array. Final shape: {data.shape}")
            
            # Convert to float64
            data = data.astype(np.float64)
            
            if self.debug:
                logger.debug(f"XISF converted: {original_shape} -> {data.shape}, dtype={data.dtype}")
                logger.debug(f"Data range: min={data.min():.2f}, max={data.max():.2f}, mean={data.mean():.2f}")
            
            # Try to extract metadata
            try:
                # Different xisf versions have different metadata access
                if hasattr(xisf_file, 'get_file_metadata'):
                    file_meta = xisf_file.get_file_metadata()
                    metadata['xisf_file_metadata'] = file_meta
                
                if hasattr(xisf_file, 'get_images_metadata'):
                    images_meta = xisf_file.get_images_metadata()
                    if images_meta and len(images_meta) > 0:
                        metadata['xisf_image_metadata'] = images_meta[0]
                
                # Try to extract FITS-like keywords from XISF metadata
                if 'xisf_image_metadata' in metadata:
                    img_meta = metadata['xisf_image_metadata']
                    
                    # Look for observation time in various formats
                    for time_key in ['DATE-OBS', 'DATE', 'FRAME']:
                        if isinstance(img_meta, dict) and time_key in img_meta:
                            try:
                                metadata['obs_time'] = Time(img_meta[time_key])
                                break
                            except:
                                continue
                    
                    # Look for WCS information
                    # XISF can store WCS in FITSKeywords
                    if isinstance(img_meta, dict) and 'FITSKeywords' in img_meta:
                        fits_keywords = img_meta['FITSKeywords']
                        
                        # Try to construct WCS from FITS keywords
                        try:
                            # Create a simple header-like dict
                            header_dict = {}
                            
                            if isinstance(fits_keywords, dict):
                                for key, value in fits_keywords.items():
                                    header_dict[key] = value
                            elif isinstance(fits_keywords, list):
                                # Sometimes it's a list of (key, value, comment) tuples
                                for item in fits_keywords:
                                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                                        header_dict[item[0]] = item[1]
                            
                            # Try to create WCS from the header dict
                            if header_dict:
                                from astropy.io.fits import Header
                                fits_header = Header()
                                for key, value in header_dict.items():
                                    try:
                                        fits_header[key] = value
                                    except:
                                        pass
                                
                                try:
                                    wcs = WCS(fits_header)
                                    metadata['wcs'] = wcs
                                except:
                                    logger.debug("Could not create WCS from XISF FITS keywords")
                        except Exception as e:
                            logger.debug(f"Error parsing XISF WCS: {e}")
                
            except Exception as e:
                logger.debug(f"Could not extract XISF metadata: {e}")
                metadata['xisf_metadata'] = "Metadata extraction not supported by this xisf version"
            
            logger.info(f"Loaded XISF image: {original_shape} -> {data.shape}")
            
            return data, metadata
            
        except ImportError:
            raise ImportError(
                "xisf library required for XISF support. Install with: pip install xisf\n"
                "For older systems, try: pip install 'xisf>=0.1.0,<1.0.0'"
            )
        except Exception as e:
            logger.error(f"Failed to load XISF file: {e}")
            raise
    
    def _load_standard_image(self, filepath: Path, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """Load standard image formats (JPG, TIFF)."""
        data = skio.imread(filepath)
        
        # Convert to grayscale if needed
        if len(data.shape) == 3:
            data = np.mean(data, axis=2)
        
        return data.astype(np.float64), metadata
    
    def preprocess_image(self, data: np.ndarray, metadata: Dict) -> np.ndarray:
        """Apply preprocessing to image data."""
        # Basic preprocessing
        processed = data.copy()
        
        # Check memory usage and potentially downsample large images
        image_size_mb = processed.nbytes / (1024 * 1024)
        if image_size_mb > 500:  # If image is larger than 500MB
            logger.warning(f"Large image detected ({image_size_mb:.1f} MB). Consider using smaller images or binning.")
        
        # Remove hot pixels and cosmic rays
        processed = self._remove_cosmic_rays(processed)
        
        # Background subtraction
        processed = self._subtract_background(processed)
        
        if self.debug:
            logger.debug(f"Image preprocessed: shape={processed.shape}, "
                        f"min={processed.min():.2f}, max={processed.max():.2f}, "
                        f"memory={processed.nbytes/(1024*1024):.1f}MB")
        
        return processed
    
    def _remove_cosmic_rays(self, data: np.ndarray) -> np.ndarray:
        """Simple cosmic ray removal using median filtering."""
        from scipy import ndimage
        median_filtered = ndimage.median_filter(data, size=3)
        diff = np.abs(data - median_filtered)
        threshold = 5 * np.std(diff)
        cosmic_rays = diff > threshold
        
        result = data.copy()
        result[cosmic_rays] = median_filtered[cosmic_rays]
        
        return result
    
    def _subtract_background(self, data: np.ndarray) -> np.ndarray:
        """Subtract background using SEP - NumPy 2.0 compatible."""
        try:
            # Ensure data is in correct byte order and type
            data_copy = np.ascontiguousarray(data, dtype=np.float64)
            
            # For NumPy 2.0+ compatibility, ensure native byte order
            if data_copy.dtype.byteorder not in ('=', '|'):
                # Convert to native byte order
                data_copy = data_copy.astype(data_copy.dtype.newbyteorder('='))
            
            bkg = sep.Background(data_copy)
            return data - bkg.back()
        except Exception as e:
            logger.warning(f"SEP background subtraction failed: {e}, using simple method")
            # Fallback to simple background subtraction
            from scipy import ndimage
            background = ndimage.gaussian_filter(data, sigma=50)
            return data - background


class StarDetector:
    """Detects and measures stars in astronomical images."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def detect_stars(self, data: np.ndarray, threshold: float = 5.0) -> Table:
        """Detect stars in the image."""
        try:
            # Ensure data is in correct format for SEP
            data_copy = np.ascontiguousarray(data, dtype=np.float64)
            
            # For NumPy 2.0+ compatibility
            if data_copy.dtype.byteorder not in ('=', '|'):
                data_copy = data_copy.astype(data_copy.dtype.newbyteorder('='))
            
            # Use SEP for source detection
            bkg = sep.Background(data_copy)
            bkg_subtracted = data_copy - bkg.back()
            
            objects = sep.extract(bkg_subtracted, threshold * bkg.globalrms)
            
            # Convert to astropy Table
            sources = Table()
            sources['x'] = objects['x']
            sources['y'] = objects['y']
            sources['flux'] = objects['flux']
            sources['a'] = objects['a']  # semi-major axis
            sources['b'] = objects['b']  # semi-minor axis
            sources['theta'] = objects['theta']
            sources['flag'] = objects['flag']
            
            # Calculate signal-to-noise ratio
            sources['snr'] = sources['flux'] / np.sqrt(sources['flux'] + bkg.globalrms**2)
            
            # Filter out likely non-stellar objects
            sources = self._filter_stellar_objects(sources, data.shape)
            
            if self.debug:
                logger.debug(f"Detected {len(sources)} stellar objects")
            
            return sources
            
        except Exception as e:
            logger.error(f"Star detection failed: {e}")
            # Fallback to DAOStarFinder
            return self._fallback_star_detection(data, threshold)
    
    def _fallback_star_detection(self, data: np.ndarray, threshold: float) -> Table:
        """Fallback star detection using photutils."""
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        daofind = DAOStarFinder(fwhm=3.0, threshold=threshold * std)
        sources = daofind(data - median)
        
        if sources is None:
            return Table()
        
        sources['snr'] = sources['peak'] / std
        return sources
    
    def _filter_stellar_objects(self, sources: Table, image_shape: Tuple) -> Table:
        """Filter to keep only stellar-like objects."""
        if len(sources) == 0:
            return sources
        
        # Remove flagged objects
        mask = sources['flag'] == 0
        
        # Remove objects that are too elongated (likely galaxies or artifacts)
        ellipticity = 1 - sources['b'] / (sources['a'] + 1e-10)  # Avoid division by zero
        mask &= ellipticity < 0.5
        
        # Remove very faint objects
        mask &= sources['snr'] > 10
        
        # Remove objects near image edges
        height, width = image_shape
        edge_buffer = 50
        mask &= (sources['x'] > edge_buffer) & (sources['x'] < width - edge_buffer)
        mask &= (sources['y'] > edge_buffer) & (sources['y'] < height - edge_buffer)
        
        return sources[mask]


class MotionDetector:
    """Detects objects with motion different from stellar motion."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.reference_stars = None
        self.transformation_matrix = None
    
    def register_images(self, image_data: List[Tuple[np.ndarray, Table, Dict]]) -> List[np.ndarray]:
        """Register images to a common reference frame."""
        if len(image_data) < 2:
            raise ValueError("Need at least 2 images for motion detection")
        
        reference_data, reference_sources, _ = image_data[0]
        registered_images = [reference_data]
        
        # Use brightest stars as reference points
        if len(reference_sources) > 0:
            n_stars = min(50, len(reference_sources))
            ref_stars = reference_sources[np.argsort(reference_sources['flux'])[-n_stars:]]
            self.reference_stars = np.column_stack([ref_stars['x'], ref_stars['y']])
        else:
            logger.warning("No reference stars found in first image")
            self.reference_stars = np.array([])
        
        for i, (data, sources, metadata) in enumerate(image_data[1:], 1):
            if self.debug:
                logger.debug(f"Registering image {i+1}/{len(image_data)}")
            
            # Find corresponding stars
            if len(sources) > 0 and len(self.reference_stars) > 0:
                n_stars = min(50, len(sources))
                curr_stars = sources[np.argsort(sources['flux'])[-n_stars:]]
                curr_positions = np.column_stack([curr_stars['x'], curr_stars['y']])
                
                # Calculate transformation
                transform = self._calculate_transformation(self.reference_stars, curr_positions)
                
                # Apply transformation
                registered = self._apply_transformation(data, transform)
                registered_images.append(registered)
            else:
                logger.warning(f"Insufficient stars for registration in image {i+1}")
                registered_images.append(data)
        
        return registered_images
    
    def _calculate_transformation(self, ref_points: np.ndarray, curr_points: np.ndarray) -> np.ndarray:
        """Calculate transformation matrix between point sets."""
        # Simple translation-based registration
        from scipy.spatial.distance import cdist
        
        distances = cdist(ref_points, curr_points)
        matches = []
        
        for i in range(min(len(ref_points), len(curr_points), 10)):
            ref_idx, curr_idx = np.unravel_index(distances.argmin(), distances.shape)
            matches.append((ref_points[ref_idx], curr_points[curr_idx]))
            distances[ref_idx, :] = np.inf
            distances[:, curr_idx] = np.inf
        
        if len(matches) < 3:
            return np.eye(3)  # Identity matrix if not enough matches
        
        # Calculate average translation
        ref_matched = np.array([m[0] for m in matches])
        curr_matched = np.array([m[1] for m in matches])
        
        translation = np.mean(ref_matched - curr_matched, axis=0)
        
        # Create transformation matrix
        transform = np.eye(3)
        transform[0:2, 2] = translation
        
        return transform
    
    def _apply_transformation(self, image: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply transformation to image."""
        # Extract translation for simple case
        translation = transform[0:2, 2]
        
        # Use OpenCV for image translation
        M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        registered = cv2.warpAffine(image.astype(np.float32), M, 
                                   (image.shape[1], image.shape[0]), 
                                   flags=cv2.INTER_LINEAR)
        
        return registered.astype(np.float64)
    
    def detect_moving_objects(self, registered_images: List[np.ndarray], 
                            source_tables: List[Table]) -> List[Dict]:
        """Detect objects that move between images."""
        moving_objects = []
        
        if len(registered_images) < 2:
            return moving_objects
        
        # Create difference images
        diff_images = []
        for i in range(1, len(registered_images)):
            diff = registered_images[i] - registered_images[0]
            diff_images.append(diff)
        
        # Detect sources in difference images
        for i, diff_img in enumerate(diff_images):
            # Smooth difference image to reduce noise
            from scipy import ndimage
            smoothed_diff = ndimage.gaussian_filter(np.abs(diff_img), sigma=1.0)
            
            # Find peaks in difference image
            threshold = 3 * np.std(smoothed_diff)
            peaks = self._find_peaks(smoothed_diff, threshold)
            
            for peak in peaks:
                x, y = peak
                
                # Verify this isn't a known star
                if not self._is_known_star(x, y, source_tables[0]):
                    obj = {
                        'x': x,
                        'y': y,
                        'image_pair': (0, i + 1),
                        'flux_diff': diff_img[int(y), int(x)],
                        'detection_time': i
                    }
                    moving_objects.append(obj)
        
        # Cluster detections that might be the same object
        if moving_objects:
            moving_objects = self._cluster_detections(moving_objects)
        
        if self.debug:
            logger.debug(f"Found {len(moving_objects)} potential moving objects")
        
        return moving_objects
    
    def _find_peaks(self, image: np.ndarray, threshold: float) -> List[Tuple[float, float]]:
        """Find peaks in image above threshold."""
        from scipy.ndimage import maximum_filter
        from scipy.ndimage import binary_erosion
        
        # Find local maxima
        local_maxima = maximum_filter(image, size=5) == image
        background = image < threshold
        eroded_background = binary_erosion(background, structure=np.ones((3, 3)))
        detected_peaks = local_maxima ^ eroded_background
        
        # Get peak coordinates
        y_coords, x_coords = np.where(detected_peaks)
        peaks = list(zip(x_coords.astype(float), y_coords.astype(float)))
        
        return peaks
    
    def _is_known_star(self, x: float, y: float, star_catalog: Table, tolerance: float = 5.0) -> bool:
        """Check if position corresponds to a known star."""
        if len(star_catalog) == 0:
            return False
        
        distances = np.sqrt((star_catalog['x'] - x)**2 + (star_catalog['y'] - y)**2)
        return np.any(distances < tolerance)
    
    def _cluster_detections(self, detections: List[Dict]) -> List[Dict]:
        """Cluster detections that likely belong to the same object."""
        if len(detections) < 2:
            return detections
        
        # Prepare data for clustering
        positions = np.array([[det['x'], det['y']] for det in detections])
        
        # Use DBSCAN to cluster nearby detections
        clustering = DBSCAN(eps=10, min_samples=1).fit(positions)
        labels = clustering.labels_
        
        # Group detections by cluster
        clustered_objects = []
        for label in set(labels):
            cluster_detections = [det for i, det in enumerate(detections) if labels[i] == label]
            
            # Calculate average position for cluster
            avg_x = np.mean([det['x'] for det in cluster_detections])
            avg_y = np.mean([det['y'] for det in cluster_detections])
            
            clustered_obj = {
                'x': avg_x,
                'y': avg_y,
                'detections': cluster_detections,
                'n_detections': len(cluster_detections)
            }
            clustered_objects.append(clustered_obj)
        
        return clustered_objects


class PlateSolver:
    """Handles plate solving for coordinate determination."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def solve_field(self, image_data: np.ndarray, metadata: Dict) -> Optional[WCS]:
        """Solve the astrometric solution for an image."""
        # If WCS already exists in metadata, use it
        if 'wcs' in metadata and metadata['wcs'] is not None:
            wcs = metadata['wcs']
            
            # Validate the WCS
            if self._validate_wcs(wcs):
                if self.debug:
                    logger.debug("Using existing WCS solution")
                return wcs
            else:
                logger.warning("WCS validation failed - WCS may be invalid or incomplete")
                return None
        
        # Try to use astrometry.net for plate solving
        try:
            return self._solve_with_astrometry_net(image_data, metadata)
        except Exception as e:
            logger.warning(f"Plate solving failed: {e}")
            return None
    
    def _validate_wcs(self, wcs: WCS) -> bool:
        """Validate that WCS has proper celestial coordinates."""
        try:
            # Check if WCS has celestial coordinates
            if not wcs.has_celestial:
                logger.warning("WCS does not have celestial coordinates")
                return False
            
            # Check coordinate types
            ctype = wcs.wcs.ctype
            if len(ctype) < 2:
                logger.warning("WCS missing coordinate types")
                return False
            
            # Check for RA/Dec coordinate system
            is_celestial = (
                ('RA' in ctype[0] or 'GLON' in ctype[0]) and
                ('DEC' in ctype[1] or 'GLAT' in ctype[1])
            )
            
            if not is_celestial:
                logger.warning(f"WCS coordinate types not recognized as celestial: {ctype}")
                return False
            
            # Try a test conversion at image center
            test_x, test_y = 512, 512  # Typical center
            try:
                world = wcs.wcs_pix2world([[test_x, test_y]], 0)
                ra, dec = float(world[0][0]), float(world[0][1])
                
                # Check if coordinates are reasonable
                if not (0 <= ra <= 360):
                    logger.warning(f"WCS test conversion gave invalid RA: {ra}")
                    return False
                if not (-90 <= dec <= 90):
                    logger.warning(f"WCS test conversion gave invalid Dec: {dec}")
                    return False
                    
            except Exception as e:
                logger.warning(f"WCS test conversion failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"WCS validation error: {e}")
            return False
    
    def _solve_with_astrometry_net(self, image_data: np.ndarray, metadata: Dict) -> Optional[WCS]:
        """Solve using astrometry.net (requires astroquery and API key)."""
        # This is a placeholder - actual implementation would require
        # astrometry.net API integration or local installation
        logger.warning("Astrometry.net integration not implemented in this example")
        return None
    
    def pixel_to_world(self, wcs: WCS, x: float, y: float) -> SkyCoord:
        """Convert pixel coordinates to world coordinates."""
        if wcs is None:
            raise ValueError("No WCS solution available")
        
        # Validate WCS first
        if not self._validate_wcs(wcs):
            raise ValueError("WCS validation failed - cannot perform coordinate conversion")
        
        try:
            # Use all_pix2world which is the most reliable method
            # This uses 0-based indexing
            world = wcs.all_pix2world([[x, y]], 0)
            ra_deg = float(world[0][0])
            dec_deg = float(world[0][1])
            
            # Validate converted coordinates
            if not (0 <= ra_deg <= 360):
                raise ValueError(f"Converted RA out of range: {ra_deg} degrees")
            if not (-90 <= dec_deg <= 90):
                raise ValueError(f"Converted Dec out of range: {dec_deg} degrees")
            
            # Create SkyCoord object
            world_coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
            
            if self.debug:
                logger.debug(f"Converted pixel ({x:.2f}, {y:.2f}) to "
                           f"RA={world_coord.ra.to_string(unit=u.hour, precision=2)}, "
                           f"Dec={world_coord.dec.to_string(unit=u.deg, precision=2)}")
            
            return world_coord
            
        except ValueError as ve:
            # Re-raise validation errors
            logger.error(f"Coordinate conversion validation failed: {ve}")
            raise
            
        except Exception as e:
            logger.error(f"Failed to convert pixel to world coordinates: {e}")
            
            # If conversion fails, provide helpful diagnostic info
            try:
                logger.info("WCS diagnostic information:")
                logger.info(f"  CTYPE: {wcs.wcs.ctype}")
                logger.info(f"  CRPIX: {wcs.wcs.crpix}")
                logger.info(f"  CRVAL: {wcs.wcs.crval}")
                logger.info(f"  CDELT: {wcs.wcs.cdelt}")
                logger.info(f"  Has celestial: {wcs.has_celestial}")
            except:
                pass
            
            raise ValueError(f"Could not convert pixel coordinates: {e}")


class ObjectIdentifier:
    """Identifies objects using astronomical databases."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def query_skybot(self, coord: SkyCoord, obs_time: Time, radius: float = 0.1, 
                     fov_width: float = None, fov_height: float = None) -> List[Dict]:
        """
        Query SkyBoT (Sky Body Tracker) service for known solar system objects.
        
        SkyBoT is an excellent service from IMCCE that searches for all known
        asteroids and comets in a given field of view at a specific time.
        
        Parameters:
        -----------
        coord : SkyCoord
            Center coordinates of the field
        obs_time : Time
            Observation time
        radius : float
            Search radius in degrees (used if fov not specified)
        fov_width : float, optional
            Field of view width in degrees
        fov_height : float, optional
            Field of view height in degrees
        
        Returns:
        --------
        List[Dict] : List of found objects with their properties
        """
        try:
            import requests
            from astropy.table import Table
            
            # If FOV not specified, use circular field with radius
            if fov_width is None:
                fov_width = radius * 2
            if fov_height is None:
                fov_height = radius * 2
            
            # SkyBoT cone search endpoint
            url = "http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php"
            
            # Prepare parameters
            params = {
                '-ra': coord.ra.deg,           # Right Ascension in degrees
                '-dec': coord.dec.deg,         # Declination in degrees
                '-rd': max(fov_width, fov_height) / 2,  # Search radius in degrees
                '-ep': obs_time.jd,            # Epoch in Julian Date
                '-loc': '500',                 # Observer location (500 = geocentric)
                '-mime': 'text',               # Response format
                '-output': 'basic'             # Output fields
            }
            
            logger.info(f"Querying SkyBoT at RA={coord.ra.deg:.4f}, Dec={coord.dec.deg:.4f}, "
                       f"JD={obs_time.jd:.2f}, radius={params['-rd']:.4f} deg")
            
            # Query SkyBoT
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            objects = []
            lines = response.text.strip().split('\n')
            
            # Skip header and comments
            data_lines = [line for line in lines if not line.startswith('#') and line.strip()]
            
            if not data_lines:
                logger.info("No objects found by SkyBoT")
                return objects
            
            # Parse each object
            for line in data_lines:
                try:
                    parts = line.split('|')
                    if len(parts) < 10:
                        continue
                    
                    # Extract object information
                    # SkyBoT format: Num|Name|RA|Dec|Type|V|PosErr|CenterDist|AngDist|Mv|Err
                    obj_number = parts[0].strip()
                    obj_name = parts[1].strip()
                    obj_ra = float(parts[2].strip())      # degrees
                    obj_dec = float(parts[3].strip())     # degrees
                    obj_type = parts[4].strip()           # Object type
                    obj_mag = parts[5].strip()            # V magnitude
                    pos_error = parts[6].strip()          # Position error in arcsec
                    
                    # Calculate separation from target coordinates
                    obj_coord = SkyCoord(ra=obj_ra*u.deg, dec=obj_dec*u.deg)
                    separation = coord.separation(obj_coord)
                    
                    # Determine object type
                    if 'C' in obj_type or 'P' in obj_type:
                        object_class = 'comet'
                    else:
                        object_class = 'asteroid'
                    
                    obj_info = {
                        'name': obj_name,
                        'number': obj_number,
                        'designation': obj_name,
                        'object_type': object_class,
                        'database': 'SkyBoT',
                        'ra': obj_ra,
                        'dec': obj_dec,
                        'separation': separation.arcsec,
                        'separation_deg': separation.deg,
                        'magnitude': obj_mag,
                        'position_error': pos_error,
                        'skybot_type': obj_type
                    }
                    
                    objects.append(obj_info)
                    
                    if self.debug:
                        logger.debug(f"Found {obj_name} ({object_class}) at "
                                   f"separation {separation.arcsec:.1f} arcsec, mag {obj_mag}")
                
                except (ValueError, IndexError) as parse_error:
                    logger.warning(f"Failed to parse SkyBoT line: {line[:50]}... Error: {parse_error}")
                    continue
            
            logger.info(f"SkyBoT found {len(objects)} objects in field")
            return objects
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SkyBoT query failed (network error): {e}")
            return []
        except Exception as e:
            logger.error(f"SkyBoT query failed: {e}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            return []
    
    def query_minor_planet_center(self, coord: SkyCoord, radius: float = 0.1) -> List[Dict]:
        """Query Minor Planet Center for known objects."""
        try:
            # MPC.query_region is not available in newer versions
            # Use SkyBoT instead as it's more comprehensive
            logger.info("MPC direct query not available - use SkyBoT for comprehensive results")
            return []
            
        except Exception as e:
            logger.error(f"MPC query failed: {e}")
            return []
    
    def query_jpl_horizons(self, coord: SkyCoord, obs_time: Time, radius: float = 0.1) -> List[Dict]:
        """Query JPL Horizons for specific known objects near coordinates."""
        try:
            from astroquery.jplhorizons import Horizons
            
            logger.info(f"Querying JPL Horizons for bright objects near coordinates")
            
            objects = []
            
            # Known bright asteroids and planets to check
            # Major asteroids
            bright_asteroids = [
                ('1', 'Ceres'),
                ('2', 'Pallas'), 
                ('3', 'Juno'),
                ('4', 'Vesta'),
                ('10', 'Hygiea'),
                ('15', 'Eunomia'),
                ('16', 'Psyche'),
                ('433', 'Eros'),
                ('624', 'Hektor'),
                ('704', 'Interamnia')
            ]
            
            # Observer location (geocentric)
            location = '500'  # Geocentric
            
            for obj_id, obj_name in bright_asteroids:
                try:
                    # Query ephemeris for this object
                    obj = Horizons(id=obj_id, location=location, epochs=obs_time.jd)
                    eph = obj.ephemerides()
                    
                    if len(eph) > 0:
                        # Get RA/Dec from ephemeris
                        obj_ra = eph['RA'][0]  # degrees
                        obj_dec = eph['DEC'][0]  # degrees
                        
                        # Calculate separation
                        obj_coord = SkyCoord(ra=obj_ra*u.deg, dec=obj_dec*u.deg)
                        separation = coord.separation(obj_coord)
                        
                        # Check if within search radius
                        if separation.deg < radius:
                            obj_info = {
                                'name': obj_name,
                                'number': obj_id,
                                'designation': f'({obj_id}) {obj_name}',
                                'object_type': 'asteroid',
                                'database': 'JPL Horizons',
                                'separation': separation.arcsec,
                                'separation_deg': separation.deg,
                                'ra': obj_ra,
                                'dec': obj_dec,
                                'magnitude': eph['V'][0] if 'V' in eph.colnames else 'N/A'
                            }
                            objects.append(obj_info)
                            logger.info(f"Found {obj_name} at separation {separation.arcsec:.1f} arcsec")
                
                except Exception as obj_error:
                    # Object not found or error - continue
                    if self.debug:
                        logger.debug(f"Could not query {obj_name}: {obj_error}")
                    continue
            
            return objects
            
        except Exception as e:
            logger.error(f"JPL Horizons query failed: {e}")
            return []
    
    def identify_object(self, coord: SkyCoord, obs_time: Optional[Time] = None, 
                       fov_radius: float = 0.1) -> Dict:
        """
        Attempt to identify an object at given coordinates.
        
        Parameters:
        -----------
        coord : SkyCoord
            Sky coordinates of detected object
        obs_time : Time, optional
            Observation time
        fov_radius : float
            Search radius in degrees (default: 0.1 deg = 6 arcmin)
        
        Returns:
        --------
        Dict : Identification results with known objects and best match
        """
        identification = {
            'coordinates': coord,
            'known_objects': [],
            'is_known': False,
            'best_match': None,
            'search_radius_deg': fov_radius,
            'search_radius_arcmin': fov_radius * 60
        }
        
        # Primary method: SkyBoT (most comprehensive)
        if obs_time:
            logger.info(f"Searching for objects at RA={coord.ra.to_string(unit=u.hour, precision=2)}, "
                       f"Dec={coord.dec.to_string(unit=u.deg, precision=2)}")
            
            try:
                skybot_results = self.query_skybot(coord, obs_time, radius=fov_radius)
                identification['known_objects'].extend(skybot_results)
                
                if skybot_results:
                    logger.info(f"SkyBoT found {len(skybot_results)} objects")
            except Exception as e:
                logger.warning(f"SkyBoT query error: {e}")
            
            # Backup method: JPL Horizons for bright objects
            try:
                horizons_results = self.query_jpl_horizons(coord, obs_time, radius=fov_radius)
                
                # Add Horizons results that aren't already in SkyBoT results
                skybot_names = {obj['name'] for obj in identification['known_objects']}
                for h_obj in horizons_results:
                    if h_obj['name'] not in skybot_names:
                        identification['known_objects'].append(h_obj)
            except Exception as e:
                logger.warning(f"JPL Horizons query error: {e}")
        else:
            logger.warning("No observation time provided - cannot query object databases")
        
        # Determine if object was identified
        if identification['known_objects']:
            identification['is_known'] = True
            
            # Sort by separation (closest first)
            identification['known_objects'].sort(
                key=lambda x: x.get('separation', float('inf'))
            )
            
            identification['best_match'] = identification['known_objects'][0]
            
            best = identification['best_match']
            logger.info(f"✓ Object identified as: {best['designation']} "
                       f"({best['object_type']}, separation: {best['separation']:.1f} arcsec)")
            
            # Log all matches if multiple found
            if len(identification['known_objects']) > 1:
                logger.info(f"  Additional {len(identification['known_objects'])-1} object(s) in field:")
                for obj in identification['known_objects'][1:4]:  # Show up to 3 more
                    logger.info(f"    - {obj['designation']} "
                              f"({obj['object_type']}, {obj['separation']:.1f} arcsec)")
        else:
            logger.info("★ Object NOT found in databases - potential NEW DISCOVERY!")
            logger.info(f"  Coordinates: RA={coord.ra.to_string(unit=u.hour, precision=2)}, "
                       f"Dec={coord.dec.to_string(unit=u.deg, precision=2)}")
            if obs_time:
                logger.info(f"  Observation time: {obs_time.iso}")
        
        return identification


class AsteroidDetector:
    """Main class coordinating the detection pipeline."""
    
    def __init__(self, debug: bool = False, progress: bool = True, max_dimension: int = 2048):
        self.debug = debug
        self.show_progress = progress
        self.max_dimension = max_dimension
        
        self.image_processor = ImageProcessor(debug, max_dimension=max_dimension)
        self.star_detector = StarDetector(debug)
        self.motion_detector = MotionDetector(debug)
        self.plate_solver = PlateSolver(debug)
        self.object_identifier = ObjectIdentifier(debug)
        
        self.results = []
    
    def process_image_sequence(self, image_paths: List[Path]) -> Dict:
        """Process a sequence of images to detect moving objects."""
        if len(image_paths) < 2:
            raise ValueError("Need at least 2 images for motion detection")
        
        logger.info(f"Processing {len(image_paths)} images")
        
        # Check total memory requirements
        total_size_mb = 0
        for path in image_paths:
            size_mb = path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
        
        logger.info(f"Total input data size: {total_size_mb:.1f} MB")
        
        if total_size_mb > 2000:  # More than 2GB of input
            logger.warning("⚠️  Large dataset detected! This may consume significant memory.")
            logger.warning("   Consider processing fewer images at once or using smaller files.")
            logger.warning(f"   Estimated peak memory usage: ~{total_size_mb * 3:.0f} MB")
        
        # Load and preprocess images
        image_data = []
        
        with tqdm(total=len(image_paths), desc="Loading images", 
                 disable=not self.show_progress) as pbar:
            for path in image_paths:
                try:
                    data, metadata = self.image_processor.load_image(path)
                    processed_data = self.image_processor.preprocess_image(data, metadata)
                    
                    # Detect stars
                    sources = self.star_detector.detect_stars(processed_data)
                    
                    # Store only what we need, delete original data to free memory
                    image_data.append((processed_data, sources, metadata))
                    del data  # Free original image memory
                    
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    if self.debug:
                        raise
                    continue
        
        if len(image_data) < 2:
            raise RuntimeError("Not enough valid images processed")
        
        # Register images
        logger.info("Registering images")
        registered_images = self.motion_detector.register_images(image_data)
        
        # Free memory from original processed images
        for i in range(len(image_data)):
            image_data[i] = (None, image_data[i][1], image_data[i][2])  # Keep only sources and metadata
        
        # Detect moving objects
        logger.info("Detecting moving objects")
        source_tables = [item[1] for item in image_data]
        moving_objects = self.motion_detector.detect_moving_objects(
            registered_images, source_tables)
        
        # Free registered images memory after detection
        del registered_images
        import gc
        gc.collect()
        
        # Plate solve and identify objects
        results = []
        wcs = None
        
        if moving_objects:
            logger.info(f"Analyzing {len(moving_objects)} potential objects")
            
            # Get WCS solution from first image (we still have metadata)
            wcs = self.plate_solver.solve_field(None, image_data[0][2])
            
            with tqdm(total=len(moving_objects), desc="Identifying objects",
                     disable=not self.show_progress) as pbar:
                for obj in moving_objects:
                    result = self._analyze_object(obj, wcs, image_data[0][2])
                    results.append(result)
                    pbar.update(1)
        
        # Compile final results
        summary = {
            'n_images_processed': len(image_data),
            'n_moving_objects_detected': len(moving_objects),
            'n_identified_objects': sum(1 for r in results if r['identification']['is_known']),
            'n_unknown_objects': sum(1 for r in results if not r['identification']['is_known']),
            'wcs_available': wcs is not None,
            'objects': results
        }
        
        self.results = summary
        return summary
    
    def _analyze_object(self, obj: Dict, wcs: Optional[WCS], metadata: Dict) -> Dict:
        """Analyze a detected moving object."""
        result = {
            'pixel_coordinates': (obj['x'], obj['y']),
            'world_coordinates': None,
            'identification': {
                'coordinates': None,
                'known_objects': [],
                'is_known': False,
                'best_match': None
            },
            'confidence': 'low',
            'metadata': obj
        }
        
        # Convert to world coordinates if possible
        if wcs is not None:
            try:
                world_coord = self.plate_solver.pixel_to_world(wcs, obj['x'], obj['y'])
                result['world_coordinates'] = world_coord
                
                # Attempt identification
                obs_time = metadata.get('obs_time')
                identification = self.object_identifier.identify_object(world_coord, obs_time)
                result['identification'] = identification
                
                # Set confidence based on number of detections and identification
                if obj.get('n_detections', 1) > 2:
                    result['confidence'] = 'high' if identification.get('is_known', False) else 'medium'
                elif obj.get('n_detections', 1) > 1:
                    result['confidence'] = 'medium' if identification.get('is_known', False) else 'low'
                
            except Exception as e:
                logger.error(f"Failed to analyze object: {e}")
                if self.debug:
                    import traceback
                    logger.debug(traceback.format_exc())
        
        return result
    
    def save_results(self, output_path: Path):
        """Save results to JSON file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Convert results to JSON-serializable format
        serializable_results = self._make_json_serializable(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, SkyCoord):
            # Convert SkyCoord to dict
            return {
                'ra_deg': float(obj.ra.deg),
                'dec_deg': float(obj.dec.deg),
                'ra_hms': obj.ra.to_string(unit=u.hour, precision=2),
                'dec_dms': obj.dec.to_string(unit=u.deg, precision=2)
            }
        elif hasattr(obj, 'to_string'):  # Other astropy objects
            return str(obj)
        elif isinstance(obj, Time):
            return obj.iso
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, u.Quantity):
            return float(obj.value)
        else:
            return obj
    
    def generate_report(self, output_path: Path):
        """Generate a detailed analysis report."""
        if not self.results:
            logger.warning("No results available for report")
            return
        
        report_lines = [
            "# Astronomical Object Detection Report",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Images processed: {self.results['n_images_processed']}",
            f"- Moving objects detected: {self.results['n_moving_objects_detected']}",
            f"- Known objects identified: {self.results['n_identified_objects']}",
            f"- Unknown objects: {self.results['n_unknown_objects']}",
            f"- Plate solving successful: {self.results['wcs_available']}",
            "",
            "## Detected Objects",
        ]
        
        if self.results['n_moving_objects_detected'] == 0:
            report_lines.extend([
                "",
                "No moving objects detected in this image sequence.",
                "",
                "This could indicate:",
                "- No asteroids/comets in the field of view during observation period",
                "- Objects too faint for detection threshold",
                "- Insufficient time separation between images",
                "- Image quality issues affecting detection",
            ])
        else:
            for i, obj in enumerate(self.results['objects'], 1):
                report_lines.extend([
                    "",
                    f"### Object {i}",
                    f"- **Pixel coordinates**: ({obj['pixel_coordinates'][0]:.1f}, {obj['pixel_coordinates'][1]:.1f})",
                    f"- **Confidence**: {obj['confidence']}",
                    f"- **Detections**: {obj['metadata'].get('n_detections', 1)}",
                ])
                
                # World coordinates
                if obj['world_coordinates']:
                    wc = obj['world_coordinates']
                    if isinstance(wc, dict):
                        report_lines.extend([
                            f"- **RA**: {wc['ra_hms']}",
                            f"- **Dec**: {wc['dec_dms']}",
                            f"- **Coordinates**: RA={wc['ra_deg']:.6f}°, Dec={wc['dec_deg']:.6f}°",
                        ])
                    else:
                        report_lines.append(f"- **Sky coordinates**: {wc}")
                
                # Identification
                if obj['identification'] and obj['identification'].get('is_known'):
                    best_match = obj['identification'].get('best_match')
                    if best_match:
                        report_lines.extend([
                            "",
                            f"#### Identification: {best_match['designation']}",
                            f"- **Object type**: {best_match['object_type']}",
                            f"- **Database**: {best_match['database']}",
                            f"- **Separation**: {best_match.get('separation', 'N/A')} arcsec",
                        ])
                        
                        if 'magnitude' in best_match and best_match['magnitude'] != 'N/A':
                            report_lines.append(f"- **Magnitude**: {best_match['magnitude']}")
                        
                        # Additional objects in field
                        other_objects = obj['identification'].get('known_objects', [])[1:4]
                        if other_objects:
                            report_lines.append("")
                            report_lines.append("**Other objects in field:**")
                            for other in other_objects:
                                report_lines.append(
                                    f"  - {other['designation']} "
                                    f"({other['object_type']}, {other.get('separation', 'N/A')} arcsec)"
                                )
                else:
                    report_lines.extend([
                        "",
                        "#### **POTENTIAL NEW DISCOVERY**",
                        "- This object was not found in known object databases",
                        "- Recommended next steps:",
                        "  1. Verify detection in original images",
                        "  2. Obtain follow-up observations",
                        "  3. Check against latest MPC circulars",
                        "  4. Consider reporting to Minor Planet Center",
                    ])
        
        # Add footer
        report_lines.extend([
            "",
            "---",
            "",
            "## Next Steps",
            "",
            "### For Identified Objects:",
            "- Cross-reference with observation predictions",
            "- Compare brightness with expected magnitude",
            "- Document any unusual behavior",
            "",
            "### For Unknown Objects:",
            "- Verify motion is consistent across all frames",
            "- Check for artifacts or image defects",
            "- Obtain additional observations for orbit determination",
            "- Search recent discovery announcements",
            "",
            "### Data Quality:",
            "- Review image alignment quality",
            "- Check for tracking errors",
            "- Verify detection threshold settings",
            "",
            f"*Report generated by Astronomical Object Detection System v1.0*"
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect moving objects in astronomical image sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s images/*.fits
  %(prog)s --debug --output results.json image1.jpg image2.jpg image3.jpg
  %(prog)s --no-progress --threshold 3.0 *.tiff
  %(prog)s --max-images 5 *.xisf  # Limit to first 5 images to save memory
        """
    )
    
    parser.add_argument('images', nargs='+', help='Input image files')
    parser.add_argument('--output', '-o', type=Path, default='detection_results.json',
                       help='Output JSON file (default: detection_results.json)')
    parser.add_argument('--report', '-r', type=Path, default='detection_report.md',
                       help='Output report file (default: detection_report.md)')
    parser.add_argument('--threshold', '-t', type=float, default=5.0,
                       help='Detection threshold (default: 5.0)')
    parser.add_argument('--max-images', '-m', type=int, default=None,
                       help='Maximum number of images to process (for memory management)')
    parser.add_argument('--max-size', type=int, default=2048,
                       help='Maximum image dimension in pixels (default: 2048, larger images will be downsampled)')
    parser.add_argument('--no-downsample', action='store_true',
                       help='Disable automatic downsampling (may use lots of memory)')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert image paths
    image_paths = [Path(p) for p in args.images]
    
    # Validate input files
    valid_paths = []
    for path in image_paths:
        if not path.exists():
            logger.error(f"File not found: {path}")
        else:
            valid_paths.append(path)
    
    if not valid_paths:
        logger.error("No valid input files found")
        sys.exit(1)
    
    # Limit number of images if specified
    if args.max_images and len(valid_paths) > args.max_images:
        logger.warning(f"Limiting to first {args.max_images} images (out of {len(valid_paths)} total)")
        logger.info("To process all images, remove --max-images flag or increase the limit")
        valid_paths = valid_paths[:args.max_images]
    
    # Check if we have enough images
    if len(valid_paths) < 2:
        logger.error("Need at least 2 images for motion detection")
        sys.exit(1)
    
    # Estimate memory usage
    total_size_mb = sum(p.stat().st_size for p in valid_paths) / (1024 * 1024)
    estimated_peak_mb = total_size_mb * 3  # Rough estimate
    
    # Adjust max dimension if downsampling is disabled
    max_dimension = None if args.no_downsample else args.max_size
    
    if args.no_downsample and estimated_peak_mb > 2000:
        logger.warning(f"⚠️  WARNING: --no-downsample specified with large files (~{estimated_peak_mb:.0f} MB)")
        logger.warning("   This will likely cause out-of-memory errors")
    elif not args.no_downsample:
        logger.info(f"Images will be downsampled to max {args.max_size}x{args.max_size} pixels to save memory")
        logger.info(f"(Use --no-downsample to disable, or --max-size to adjust)")
    
    if estimated_peak_mb > 8000 and args.no_downsample:  # More than 8GB
        logger.error(f"❌ Cannot process: Estimated memory usage: ~{estimated_peak_mb:.0f} MB")
        logger.error("   Use --max-size 2048 (or smaller) to enable downsampling")
        sys.exit(1)
    
    try:
        # Create detector with memory-saving settings
        detector = AsteroidDetector(
            debug=args.debug, 
            progress=not args.no_progress,
            max_dimension=max_dimension
        )
        
        # Process images
        results = detector.process_image_sequence(valid_paths)
        
        # Save results
        detector.save_results(args.output)
        detector.generate_report(args.report)
        
        # Print summary
        print(f"\n=== Detection Summary ===")
        print(f"Images processed: {results['n_images_processed']}")
        print(f"Moving objects detected: {results['n_moving_objects_detected']}")
        print(f"Known objects identified: {results['n_identified_objects']}")
        print(f"Unknown objects: {results['n_unknown_objects']}")
        
        if results['n_unknown_objects'] > 0:
            print(f"\n🎉 Found {results['n_unknown_objects']} potential new discoveries!")
        
        print(f"\nResults saved to: {args.output}")
        print(f"Report saved to: {args.report}")
        
    except MemoryError:
        logger.error("❌ Out of memory error!")
        logger.error("Your images are too large to process all at once.")
        logger.error("Try one of these solutions:")
        logger.error("  1. Process fewer images: python asteroid_detector.py --max-images 3 *.xisf")
        logger.error("  2. Bin your images 2x2 in PixInsight before detection")
        logger.error("  3. Convert to smaller FITS files")
        logger.error("  4. Run on a machine with more RAM")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()