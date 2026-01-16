#!/usr/bin/env python3
"""
Astronomical Object Detection System
=====================================
Detects moving objects (asteroids, comets) in astronomical image sequences
by identifying objects that don't follow stellar motion patterns.

Author: Dennis
License: MIT
"""

import argparse
import logging
import os
import sys
import time
import json
import struct
import zlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Any, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray
import requests

# Astronomy libraries
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
from astroquery.jplhorizons import Horizons
from astroquery.mpc import MPC

# Image processing
import sep
from skimage import io as skio
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
from sklearn.cluster import DBSCAN
import cv2
from PIL import Image

# Progress indication
from tqdm import tqdm

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ImageMetadata:
    """Metadata extracted from astronomical images."""
    filepath: Path
    width: int
    height: int
    observation_time: Optional[datetime] = None
    exposure_time: Optional[float] = None
    filter_name: Optional[str] = None
    telescope: Optional[str] = None
    observer: Optional[str] = None
    ra_center: Optional[float] = None
    dec_center: Optional[float] = None
    pixel_scale: Optional[float] = None
    wcs: Optional[WCS] = None
    header: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedSource:
    """A detected source in an image."""
    x: float
    y: float
    flux: float
    snr: float
    fwhm: float
    ellipticity: float
    ra: Optional[float] = None
    dec: Optional[float] = None
    magnitude: Optional[float] = None


@dataclass  
class MovingObject:
    """A detected moving object across multiple frames."""
    id: str
    positions: List[Tuple[float, float]]
    times: List[datetime]
    ra_positions: List[float]
    dec_positions: List[float]
    velocity_arcsec_per_hour: float
    position_angle: float
    confidence: float
    matched_name: Optional[str] = None
    matched_designation: Optional[str] = None
    is_known: bool = False
    orbital_elements: Optional[Dict] = None


@dataclass
class DetectionResult:
    """Complete detection result for an image sequence."""
    input_files: List[str]
    processing_time: float
    sources_per_image: List[int]
    moving_objects: List[MovingObject]
    potential_discoveries: List[MovingObject]
    known_objects: List[MovingObject]
    errors: List[str]
    warnings: List[str]


# =============================================================================
# Progress Tracking
# =============================================================================

class ProgressTracker:
    """Thread-safe progress tracking with multiple output modes."""
    
    def __init__(self, total: int, description: str = "", 
                 mode: str = "tqdm", callback: Optional[Callable] = None):
        self.total = total
        self.current = 0
        self.description = description
        self.mode = mode
        self.callback = callback
        self._pbar = None
        
        if mode == "tqdm":
            self._pbar = tqdm(total=total, desc=description, unit="step")
    
    def update(self, n: int = 1, status: str = ""):
        """Update progress by n steps."""
        self.current += n
        
        if self.mode == "tqdm" and self._pbar:
            self._pbar.update(n)
            if status:
                self._pbar.set_postfix_str(status)
        elif self.mode == "log":
            pct = (self.current / self.total) * 100
            logger.info(f"[{pct:.1f}%] {self.description}: {status}")
        elif self.mode == "callback" and self.callback:
            self.callback(self.current, self.total, status)
    
    def close(self):
        if self._pbar:
            self._pbar.close()


@contextmanager
def progress_context(total: int, description: str, mode: str = "tqdm"):
    """Context manager for progress tracking."""
    tracker = ProgressTracker(total, description, mode)
    try:
        yield tracker
    finally:
        tracker.close()


# =============================================================================
# Image Processing
# =============================================================================

class XISFReader:
    """Reader for XISF (Extensible Image Serialization Format) files."""
    
    SIGNATURE = b'XISF0100'
    
    @classmethod
    def read(cls, filepath: Path) -> Tuple[NDArray, Dict[str, Any]]:
        """Read an XISF file and return image data and metadata."""
        with open(filepath, 'rb') as f:
            # Read and validate signature
            sig = f.read(8)
            if sig != cls.SIGNATURE:
                raise ValueError(f"Invalid XISF signature: {sig}")
            
            # Read header length
            header_len = struct.unpack('<I', f.read(4))[0]
            reserved = f.read(4)  # Reserved bytes
            
            # Read XML header
            header_xml = f.read(header_len).decode('utf-8')
            
            # Parse XML to extract image properties
            import xml.etree.ElementTree as ET
            root = ET.fromstring(header_xml)
            
            # Find image element
            ns = {'xisf': 'http://www.pixinsight.com/xisf'}
            image_elem = root.find('.//xisf:Image', ns)
            if image_elem is None:
                # Try without namespace
                image_elem = root.find('.//Image')
            
            if image_elem is None:
                raise ValueError("No Image element found in XISF header")
            
            # Extract geometry
            geometry = image_elem.get('geometry', '').split(':')
            if len(geometry) >= 2:
                width, height = int(geometry[0]), int(geometry[1])
                channels = int(geometry[2]) if len(geometry) > 2 else 1
            else:
                raise ValueError("Invalid geometry in XISF")
            
            # Get data type
            sample_format = image_elem.get('sampleFormat', 'Float32')
            dtype_map = {
                'UInt8': np.uint8,
                'UInt16': np.uint16,
                'UInt32': np.uint32,
                'Float32': np.float32,
                'Float64': np.float64
            }
            dtype = dtype_map.get(sample_format, np.float32)
            
            # Get data location
            location = image_elem.get('location', '')
            parts = location.split(':')
            
            if parts[0] == 'attachment':
                # Data is attached after header
                offset = int(parts[1]) if len(parts) > 1 else 0
                size = int(parts[2]) if len(parts) > 2 else None
                
                f.seek(16 + header_len + offset)
                
                if size:
                    raw_data = f.read(size)
                else:
                    raw_data = f.read()
                
                # Check for compression
                compression = image_elem.get('compression', '')
                if compression.startswith('zlib'):
                    raw_data = zlib.decompress(raw_data)
                
                # Convert to numpy array
                data = np.frombuffer(raw_data, dtype=dtype)
                
                if channels > 1:
                    data = data.reshape((channels, height, width))
                    # Convert to grayscale if color
                    if channels == 3:
                        data = np.mean(data, axis=0)
                    else:
                        data = data[0]
                else:
                    data = data.reshape((height, width))
                
            else:
                raise ValueError(f"Unsupported XISF location type: {parts[0]}")
            
            # Extract metadata
            metadata = {
                'width': width,
                'height': height,
                'sample_format': sample_format
            }
            
            # Parse FITS keywords if present
            for prop in root.findall('.//xisf:FITSKeyword', ns) + root.findall('.//FITSKeyword'):
                name = prop.get('name', '')
                value = prop.get('value', '')
                metadata[name] = value
            
            return data.astype(np.float32), metadata


class ImageProcessor:
    """Handles loading and preprocessing of astronomical images."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.tiff', '.tif', '.fits', '.fit', '.fts', '.xisf'}
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def load_image(self, filepath: Path) -> Tuple[NDArray, ImageMetadata]:
        """Load an astronomical image and extract metadata."""
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}. Supported: {self.SUPPORTED_FORMATS}")
        
        if suffix in {'.fits', '.fit', '.fts'}:
            return self._load_fits(filepath)
        elif suffix == '.xisf':
            return self._load_xisf(filepath)
        elif suffix in {'.tiff', '.tif'}:
            return self._load_tiff(filepath)
        else:  # JPEG
            return self._load_jpeg(filepath)
    
    def _load_fits(self, filepath: Path) -> Tuple[NDArray, ImageMetadata]:
        """Load a FITS file."""
        with fits.open(filepath) as hdul:
            # Find the image HDU
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    data = hdu.data.astype(np.float32)
                    header = dict(hdu.header)
                    break
            else:
                raise ValueError("No image data found in FITS file")
            
            # Handle 3D data (color or cube)
            if data.ndim == 3:
                if data.shape[0] == 3:  # RGB
                    data = np.mean(data, axis=0)
                else:
                    data = data[0]
            
            # Extract WCS if present
            try:
                wcs = WCS(hdu.header)
                if not wcs.has_celestial:
                    wcs = None
            except Exception:
                wcs = None
            
            # Parse observation time
            obs_time = None
            for key in ['DATE-OBS', 'DATE_OBS', 'MJD-OBS']:
                if key in header:
                    try:
                        if key == 'MJD-OBS':
                            obs_time = Time(header[key], format='mjd').datetime
                        else:
                            obs_time = Time(header[key], format='isot').datetime
                        break
                    except Exception:
                        continue
            
            metadata = ImageMetadata(
                filepath=filepath,
                width=data.shape[1],
                height=data.shape[0],
                observation_time=obs_time,
                exposure_time=header.get('EXPTIME') or header.get('EXPOSURE'),
                filter_name=header.get('FILTER'),
                telescope=header.get('TELESCOP'),
                observer=header.get('OBSERVER'),
                ra_center=header.get('CRVAL1'),
                dec_center=header.get('CRVAL2'),
                pixel_scale=header.get('CDELT1') or header.get('PIXSCALE'),
                wcs=wcs,
                header=header
            )
            
            return data, metadata
    
    def _load_xisf(self, filepath: Path) -> Tuple[NDArray, ImageMetadata]:
        """Load an XISF file."""
        data, xisf_meta = XISFReader.read(filepath)
        
        # Parse observation time from FITS keywords
        obs_time = None
        for key in ['DATE-OBS', 'DATE_OBS']:
            if key in xisf_meta:
                try:
                    obs_time = Time(xisf_meta[key], format='isot').datetime
                    break
                except Exception:
                    continue
        
        metadata = ImageMetadata(
            filepath=filepath,
            width=data.shape[1],
            height=data.shape[0],
            observation_time=obs_time,
            exposure_time=float(xisf_meta.get('EXPTIME', 0)) or None,
            filter_name=xisf_meta.get('FILTER'),
            telescope=xisf_meta.get('TELESCOP'),
            header=xisf_meta
        )
        
        return data, metadata
    
    def _load_tiff(self, filepath: Path) -> Tuple[NDArray, ImageMetadata]:
        """Load a TIFF file."""
        data = skio.imread(str(filepath))
        
        # Convert to grayscale if color
        if data.ndim == 3:
            if data.shape[2] == 3:
                data = np.mean(data, axis=2)
            elif data.shape[2] == 4:  # RGBA
                data = np.mean(data[:, :, :3], axis=2)
        
        data = data.astype(np.float32)
        
        metadata = ImageMetadata(
            filepath=filepath,
            width=data.shape[1],
            height=data.shape[0]
        )
        
        return data, metadata
    
    def _load_jpeg(self, filepath: Path) -> Tuple[NDArray, ImageMetadata]:
        """Load a JPEG file."""
        img = Image.open(filepath)
        data = np.array(img.convert('L'), dtype=np.float32)
        
        metadata = ImageMetadata(
            filepath=filepath,
            width=data.shape[1],
            height=data.shape[0]
        )
        
        return data, metadata
    
    def preprocess(self, data: NDArray, 
                   subtract_background: bool = True,
                   remove_hot_pixels: bool = True) -> NDArray:
        """Preprocess image data for detection."""
        result = data.copy()
        
        # Ensure proper byte order for SEP
        if not result.flags['C_CONTIGUOUS']:
            result = np.ascontiguousarray(result)
        
        # Remove hot pixels using median filter
        if remove_hot_pixels:
            from scipy.ndimage import median_filter
            filtered = median_filter(result, size=3)
            # Only replace extreme outliers
            threshold = np.std(result) * 5
            mask = np.abs(result - filtered) > threshold
            result[mask] = filtered[mask]
        
        # Background subtraction using SEP
        if subtract_background:
            try:
                bkg = sep.Background(result.byteswap().newbyteorder())
                result = result - bkg.back()
            except Exception as e:
                logger.warning(f"SEP background subtraction failed: {e}, using simple method")
                result = result - np.median(result)
        
        return result


# =============================================================================
# Star Detection
# =============================================================================

class StarDetector:
    """Detects stellar sources in astronomical images."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def detect(self, data: NDArray, threshold: float = 3.0,
               min_area: int = 5, deblend: bool = True) -> List[DetectedSource]:
        """Detect sources in the image using SEP."""
        try:
            # Ensure proper format for SEP
            data_sep = data.byteswap().newbyteorder()
            
            # Background estimation
            bkg = sep.Background(data_sep)
            data_sub = data - bkg.back()
            
            # Source extraction
            objects = sep.extract(
                data_sub.byteswap().newbyteorder(),
                threshold,
                err=bkg.globalrms,
                minarea=min_area,
                deblend_cont=0.005 if deblend else 1.0,
                deblend_nthresh=32 if deblend else 1
            )
            
            if self.debug:
                logger.debug(f"SEP extracted {len(objects)} raw objects")
            
            # Convert to DetectedSource objects
            sources = []
            for obj in objects:
                # Calculate FWHM from semi-axes
                fwhm = 2.35 * np.sqrt((obj['a']**2 + obj['b']**2) / 2)
                
                # Calculate ellipticity
                if obj['a'] > 0:
                    ellipticity = 1 - obj['b'] / obj['a']
                else:
                    ellipticity = 0
                
                # Calculate SNR
                snr = obj['flux'] / (bkg.globalrms * np.sqrt(obj['npix']))
                
                source = DetectedSource(
                    x=float(obj['x']),
                    y=float(obj['y']),
                    flux=float(obj['flux']),
                    snr=float(snr),
                    fwhm=float(fwhm),
                    ellipticity=float(ellipticity)
                )
                sources.append(source)
            
            # Filter sources
            sources = self._filter_sources(sources, data.shape)
            
            if self.debug:
                logger.debug(f"After filtering: {len(sources)} sources")
            
            return sources
            
        except Exception as e:
            logger.error(f"SEP detection failed: {e}")
            return self._fallback_detection(data, threshold)
    
    def _filter_sources(self, sources: List[DetectedSource], 
                        shape: Tuple[int, int]) -> List[DetectedSource]:
        """Filter detected sources to remove artifacts."""
        height, width = shape
        edge_buffer = 20
        
        filtered = []
        for src in sources:
            # Skip edge sources
            if src.x < edge_buffer or src.x > width - edge_buffer:
                continue
            if src.y < edge_buffer or src.y > height - edge_buffer:
                continue
            
            # Skip very elongated objects (likely cosmic rays or satellites)
            if src.ellipticity > 0.7:
                continue
            
            # Skip very faint sources
            if src.snr < 5:
                continue
            
            # Skip sources with unrealistic FWHM
            if src.fwhm < 1 or src.fwhm > 50:
                continue
            
            filtered.append(src)
        
        return filtered
    
    def _fallback_detection(self, data: NDArray, threshold: float) -> List[DetectedSource]:
        """Fallback detection using simple peak finding."""
        from scipy.ndimage import maximum_filter, label
        
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        threshold_val = median + threshold * std
        
        # Find local maxima
        local_max = maximum_filter(data, size=5) == data
        peaks = (data > threshold_val) & local_max
        
        labeled, num_features = label(peaks)
        
        sources = []
        for i in range(1, num_features + 1):
            y_coords, x_coords = np.where(labeled == i)
            if len(x_coords) == 0:
                continue
            
            x = float(np.mean(x_coords))
            y = float(np.mean(y_coords))
            flux = float(data[int(y), int(x)])
            snr = float((flux - median) / std)
            
            sources.append(DetectedSource(
                x=x, y=y, flux=flux, snr=snr, fwhm=3.0, ellipticity=0.0
            ))
        
        return sources


# =============================================================================
# Image Registration
# =============================================================================

class ImageRegistrar:
    """Aligns images to a common reference frame."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def compute_shift(self, reference: NDArray, target: NDArray) -> Tuple[float, float]:
        """Compute shift between two images using cross-correlation."""
        # Use phase cross-correlation for sub-pixel accuracy
        shift, error, diffphase = phase_cross_correlation(
            reference, target, upsample_factor=10
        )
        
        if self.debug:
            logger.debug(f"Computed shift: dy={shift[0]:.2f}, dx={shift[1]:.2f}")
        
        return float(shift[1]), float(shift[0])  # Return as (dx, dy)
    
    def compute_transform(self, ref_sources: List[DetectedSource],
                          target_sources: List[DetectedSource],
                          max_dist: float = 10.0) -> Optional[AffineTransform]:
        """Compute affine transform by matching sources."""
        if len(ref_sources) < 3 or len(target_sources) < 3:
            return None
        
        # Extract positions
        ref_pos = np.array([(s.x, s.y) for s in ref_sources])
        tgt_pos = np.array([(s.x, s.y) for s in target_sources])
        
        # Match sources using triangles (simplified approach)
        # Build KD-tree for fast nearest neighbor lookup
        from scipy.spatial import cKDTree
        
        # Use brightest sources for matching
        ref_sorted = sorted(ref_sources, key=lambda s: s.flux, reverse=True)[:50]
        tgt_sorted = sorted(target_sources, key=lambda s: s.flux, reverse=True)[:50]
        
        ref_pos = np.array([(s.x, s.y) for s in ref_sorted])
        tgt_pos = np.array([(s.x, s.y) for s in tgt_sorted])
        
        # Initial shift estimate from cross-correlation
        # (would need image data, so skip for now)
        
        # Try to find matching pairs
        tree = cKDTree(tgt_pos)
        
        matched_ref = []
        matched_tgt = []
        
        for i, pos in enumerate(ref_pos):
            dist, idx = tree.query(pos, k=1)
            if dist < max_dist:
                matched_ref.append(pos)
                matched_tgt.append(tgt_pos[idx])
        
        if len(matched_ref) < 3:
            return None
        
        matched_ref = np.array(matched_ref)
        matched_tgt = np.array(matched_tgt)
        
        # Estimate affine transform using RANSAC
        from skimage.measure import ransac
        
        try:
            model, inliers = ransac(
                (matched_tgt, matched_ref),
                AffineTransform,
                min_samples=3,
                residual_threshold=2,
                max_trials=1000
            )
            
            if self.debug:
                logger.debug(f"Transform found with {np.sum(inliers)} inliers")
            
            return model
        except Exception as e:
            logger.warning(f"Transform estimation failed: {e}")
            return None
    
    def align_image(self, image: NDArray, transform: AffineTransform) -> NDArray:
        """Apply transform to align image."""
        aligned = warp(image, transform.inverse, preserve_range=True)
        return aligned.astype(np.float32)


# =============================================================================
# Motion Detection
# =============================================================================

class MotionDetector:
    """Detects objects with motion different from stellar motion."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.min_detections = 3  # Minimum frames to confirm detection
    
    def detect_motion(self, images: List[NDArray],
                      sources_list: List[List[DetectedSource]],
                      times: List[datetime],
                      stellar_shift: Optional[Tuple[float, float]] = None
                     ) -> List[MovingObject]:
        """Detect objects moving differently from stars."""
        
        if len(images) < 2:
            raise ValueError("Need at least 2 images for motion detection")
        
        # Step 1: Compute expected stellar motion between frames
        registrar = ImageRegistrar(debug=self.debug)
        stellar_shifts = []
        
        for i in range(1, len(images)):
            dx, dy = registrar.compute_shift(images[0], images[i])
            stellar_shifts.append((dx, dy))
        
        if self.debug:
            logger.debug(f"Stellar shifts: {stellar_shifts}")
        
        # Step 2: Find sources that don't follow stellar motion
        candidates = self._find_non_stellar_motion(
            sources_list, stellar_shifts, times
        )
        
        if self.debug:
            logger.debug(f"Found {len(candidates)} motion candidates")
        
        # Step 3: Link detections across frames
        moving_objects = self._link_detections(candidates, times)
        
        # Step 4: Filter false positives
        moving_objects = self._filter_false_positives(moving_objects)
        
        return moving_objects
    
    def _find_non_stellar_motion(self, sources_list: List[List[DetectedSource]],
                                  stellar_shifts: List[Tuple[float, float]],
                                  times: List[datetime]
                                 ) -> List[Dict]:
        """Find sources that don't match stellar motion."""
        candidates = []
        
        # Use first frame as reference
        ref_sources = sources_list[0]
        
        for frame_idx in range(1, len(sources_list)):
            dx_stellar, dy_stellar = stellar_shifts[frame_idx - 1]
            curr_sources = sources_list[frame_idx]
            
            # Predict where each reference source should be
            for ref_src in ref_sources:
                predicted_x = ref_src.x + dx_stellar
                predicted_y = ref_src.y + dy_stellar
                
                # Find closest match in current frame
                best_match = None
                best_dist = float('inf')
                
                for curr_src in curr_sources:
                    dist = np.sqrt((curr_src.x - predicted_x)**2 + 
                                   (curr_src.y - predicted_y)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = curr_src
                
                # If no good match, this source moved differently
                if best_dist > 5.0:  # More than 5 pixels from prediction
                    # Look for the actual match (non-stellar motion)
                    for curr_src in curr_sources:
                        dist_from_ref = np.sqrt((curr_src.x - ref_src.x)**2 + 
                                                (curr_src.y - ref_src.y)**2)
                        # Must have moved, but not too far
                        if 2.0 < dist_from_ref < 200.0:
                            candidates.append({
                                'frame': frame_idx,
                                'ref_pos': (ref_src.x, ref_src.y),
                                'curr_pos': (curr_src.x, curr_src.y),
                                'flux': curr_src.flux,
                                'time': times[frame_idx]
                            })
        
        return candidates
    
    def _link_detections(self, candidates: List[Dict],
                         times: List[datetime]) -> List[MovingObject]:
        """Link detections across frames into tracklets."""
        if not candidates:
            return []
        
        # Extract positions for clustering
        positions = np.array([c['curr_pos'] for c in candidates])
        
        if len(positions) < 2:
            return []
        
        # Use DBSCAN to cluster nearby detections
        clustering = DBSCAN(eps=20, min_samples=2).fit(positions)
        
        moving_objects = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Noise
                continue
            
            mask = clustering.labels_ == label
            cluster_candidates = [c for c, m in zip(candidates, mask) if m]
            
            if len(cluster_candidates) < self.min_detections:
                continue
            
            # Sort by frame number
            cluster_candidates.sort(key=lambda c: c['frame'])
            
            # Calculate motion parameters
            positions_list = [c['curr_pos'] for c in cluster_candidates]
            times_list = [c['time'] for c in cluster_candidates]
            
            # Calculate velocity
            if len(times_list) >= 2:
                dt = (times_list[-1] - times_list[0]).total_seconds() / 3600.0  # hours
                if dt > 0:
                    dx = positions_list[-1][0] - positions_list[0][0]
                    dy = positions_list[-1][1] - positions_list[0][1]
                    
                    # Assume ~1 arcsec/pixel (should use actual plate scale)
                    velocity = np.sqrt(dx**2 + dy**2) / dt
                    angle = np.degrees(np.arctan2(dy, dx))
                    
                    # Calculate confidence based on number of detections and consistency
                    confidence = min(1.0, len(cluster_candidates) / 5.0)
                    
                    obj = MovingObject(
                        id=f"obj_{len(moving_objects):04d}",
                        positions=positions_list,
                        times=times_list,
                        ra_positions=[],
                        dec_positions=[],
                        velocity_arcsec_per_hour=velocity,
                        position_angle=angle,
                        confidence=confidence
                    )
                    moving_objects.append(obj)
        
        return moving_objects
    
    def _filter_false_positives(self, objects: List[MovingObject]) -> List[MovingObject]:
        """Filter out likely false positive detections."""
        filtered = []
        
        for obj in objects:
            # Skip if too few detections
            if len(obj.positions) < self.min_detections:
                continue
            
            # Skip if motion is too fast (likely satellite or cosmic ray)
            if obj.velocity_arcsec_per_hour > 1000:  # arcsec/hour
                if self.debug:
                    logger.debug(f"Filtering {obj.id}: velocity too high")
                continue
            
            # Skip if motion is too slow (likely star or noise)
            if obj.velocity_arcsec_per_hour < 0.5:
                if self.debug:
                    logger.debug(f"Filtering {obj.id}: velocity too low")
                continue
            
            # Check for linear motion (asteroids move linearly over short times)
            if not self._is_linear_motion(obj.positions):
                if self.debug:
                    logger.debug(f"Filtering {obj.id}: non-linear motion")
                continue
            
            filtered.append(obj)
        
        return filtered
    
    def _is_linear_motion(self, positions: List[Tuple[float, float]], 
                          tolerance: float = 3.0) -> bool:
        """Check if positions follow approximately linear motion."""
        if len(positions) < 3:
            return True
        
        # Fit a line to positions
        x = np.array([p[0] for p in positions])
        y = np.array([p[1] for p in positions])
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        y_fit = np.polyval(coeffs, x)
        
        # Calculate residuals
        residuals = np.abs(y - y_fit)
        
        return np.max(residuals) < tolerance


# =============================================================================
# Plate Solving
# =============================================================================

class PlateSolver:
    """Determines celestial coordinates from image."""
    
    ASTROMETRY_NET_URL = "http://nova.astrometry.net/api"
    
    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        self.api_key = api_key
        self.debug = debug
        self._session_key = None
    
    def solve_from_wcs(self, wcs: WCS, 
                       pixel_coords: List[Tuple[float, float]]
                      ) -> List[Tuple[float, float]]:
        """Convert pixel coordinates to celestial using existing WCS."""
        if wcs is None:
            return []
        
        celestial_coords = []
        for x, y in pixel_coords:
            try:
                sky = wcs.pixel_to_world(x, y)
                celestial_coords.append((sky.ra.deg, sky.dec.deg))
            except Exception:
                celestial_coords.append((None, None))
        
        return celestial_coords
    
    def solve_online(self, image: NDArray,
                     sources: List[DetectedSource],
                     timeout: int = 300) -> Optional[WCS]:
        """Solve plate using astrometry.net API."""
        if not self.api_key:
            logger.warning("No astrometry.net API key provided")
            return None
        
        try:
            # Login to get session
            self._login()
            
            # Upload image
            submission_id = self._upload_image(image)
            
            # Wait for solution
            job_id = self._wait_for_submission(submission_id, timeout)
            
            if job_id:
                wcs = self._get_solution(job_id)
                return wcs
            
        except Exception as e:
            logger.error(f"Online plate solving failed: {e}")
        
        return None
    
    def _login(self):
        """Login to astrometry.net API."""
        response = requests.post(
            f"{self.ASTROMETRY_NET_URL}/login",
            data={'request-json': json.dumps({'apikey': self.api_key})}
        )
        result = response.json()
        if result.get('status') == 'success':
            self._session_key = result['session']
        else:
            raise RuntimeError(f"API login failed: {result}")
    
    def _upload_image(self, image: NDArray) -> int:
        """Upload image for solving."""
        # Save as temporary FITS
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            hdu = fits.PrimaryHDU(image)
            hdu.writeto(f.name, overwrite=True)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                response = requests.post(
                    f"{self.ASTROMETRY_NET_URL}/upload",
                    files={'file': f},
                    data={'request-json': json.dumps({
                        'session': self._session_key,
                        'allow_commercial_use': 'n',
                        'allow_modifications': 'n'
                    })}
                )
            result = response.json()
            if result.get('status') == 'success':
                return result['subid']
            else:
                raise RuntimeError(f"Upload failed: {result}")
        finally:
            os.unlink(temp_path)
    
    def _wait_for_submission(self, submission_id: int, timeout: int) -> Optional[int]:
        """Wait for submission to be processed."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self.ASTROMETRY_NET_URL}/submissions/{submission_id}"
            )
            result = response.json()
            
            jobs = result.get('jobs', [])
            if jobs:
                job_id = jobs[0]
                if job_id:
                    # Check job status
                    job_response = requests.get(
                        f"{self.ASTROMETRY_NET_URL}/jobs/{job_id}"
                    )
                    job_result = job_response.json()
                    
                    if job_result.get('status') == 'success':
                        return job_id
                    elif job_result.get('status') == 'failure':
                        logger.warning("Plate solving failed")
                        return None
            
            time.sleep(5)
        
        logger.warning("Plate solving timed out")
        return None
    
    def _get_solution(self, job_id: int) -> Optional[WCS]:
        """Get WCS solution from completed job."""
        response = requests.get(
            f"{self.ASTROMETRY_NET_URL}/jobs/{job_id}/wcs_file"
        )
        
        if response.status_code == 200:
            # Parse WCS from returned FITS header
            from io import BytesIO
            with fits.open(BytesIO(response.content)) as hdul:
                return WCS(hdul[0].header)
        
        return None


# =============================================================================
# Object Identification
# =============================================================================

class ObjectIdentifier:
    """Identifies detected objects using astronomical databases."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def identify(self, obj: MovingObject, 
                 search_radius: float = 60.0) -> MovingObject:
        """Try to identify a moving object using databases."""
        if not obj.ra_positions or not obj.dec_positions:
            return obj
        
        # Use middle position for search
        mid_idx = len(obj.ra_positions) // 2
        ra = obj.ra_positions[mid_idx]
        dec = obj.dec_positions[mid_idx]
        obs_time = obj.times[mid_idx]
        
        # Try JPL Horizons first
        match = self._search_jpl_horizons(ra, dec, obs_time, search_radius)
        
        if match:
            obj.matched_name = match.get('name')
            obj.matched_designation = match.get('designation')
            obj.is_known = True
            obj.orbital_elements = match.get('orbital_elements')
            return obj
        
        # Try Minor Planet Center
        match = self._search_mpc(ra, dec, obs_time, search_radius)
        
        if match:
            obj.matched_name = match.get('name')
            obj.matched_designation = match.get('designation')
            obj.is_known = True
            return obj
        
        # No match found - potential new discovery
        obj.is_known = False
        return obj
    
    def _search_jpl_horizons(self, ra: float, dec: float,
                              obs_time: datetime,
                              radius: float) -> Optional[Dict]:
        """Search JPL Horizons for known objects near position."""
        try:
            # Convert time to Horizons format
            time_str = obs_time.strftime('%Y-%m-%d %H:%M')
            
            # This is a simplified search - real implementation would
            # query the Small Body Database API
            url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
            params = {
                'fields': 'spkid,full_name,kind,e,a,i,om,w,ma,epoch',
                'sb-kind': 'a',  # Asteroids
                'limit': 10
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                # Check each object's ephemeris
                # (Simplified - real version would compute ephemerides)
                
                if self.debug:
                    logger.debug(f"JPL query returned {len(data.get('data', []))} objects")
                
                # For now, return None - real implementation needs ephemeris calculation
                return None
            
        except Exception as e:
            logger.warning(f"JPL Horizons search failed: {e}")
        
        return None
    
    def _search_mpc(self, ra: float, dec: float,
                    obs_time: datetime,
                    radius: float) -> Optional[Dict]:
        """Search Minor Planet Center for known objects."""
        try:
            # Query MPC using astroquery
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            
            # Search for known objects at this position
            result = MPC.query_object('minor_planet', 
                                      target_type='asteroid',
                                      get_query_payload=True)
            
            if self.debug:
                logger.debug(f"MPC query for RA={ra:.4f}, Dec={dec:.4f}")
            
            # For demonstration - real implementation would use
            # the MPC's conesearch or ephemeris services
            return None
            
        except Exception as e:
            logger.warning(f"MPC search failed: {e}")
        
        return None
    
    def check_known_object_at_position(self, ra: float, dec: float,
                                        obs_time: datetime,
                                        target_name: str) -> bool:
        """Check if a specific known object is at the given position."""
        try:
            # Query Horizons for specific object
            obj = Horizons(
                id=target_name,
                location='500',  # Geocentric
                epochs=Time(obs_time).jd
            )
            
            eph = obj.ephemerides()
            
            if len(eph) > 0:
                obj_ra = eph['RA'][0]
                obj_dec = eph['DEC'][0]
                
                # Check if within 60 arcsec
                sep = np.sqrt((ra - obj_ra)**2 + (dec - obj_dec)**2) * 3600
                
                return sep < 60
            
        except Exception as e:
            if self.debug:
                logger.debug(f"Horizons check for {target_name} failed: {e}")
        
        return False


# =============================================================================
# Main Detector Class
# =============================================================================

class AsteroidDetector:
    """Main class for asteroid detection pipeline."""
    
    def __init__(self, 
                 debug: bool = False,
                 verbose: bool = False,
                 progress_mode: str = "tqdm",
                 astrometry_api_key: Optional[str] = None):
        """
        Initialize the asteroid detector.
        
        Args:
            debug: Enable debug logging
            verbose: Enable verbose output
            progress_mode: Progress display mode ("tqdm", "log", "callback", "none")
            astrometry_api_key: API key for astrometry.net plate solving
        """
        self.debug = debug
        self.verbose = verbose
        self.progress_mode = progress_mode
        
        # Configure logging
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        elif verbose:
            logging.getLogger().setLevel(logging.INFO)
        
        # Initialize components
        self.image_processor = ImageProcessor(debug=debug)
        self.star_detector = StarDetector(debug=debug)
        self.motion_detector = MotionDetector(debug=debug)
        self.plate_solver = PlateSolver(api_key=astrometry_api_key, debug=debug)
        self.object_identifier = ObjectIdentifier(debug=debug)
    
    def detect(self, image_paths: List[Path],
               detection_threshold: float = 3.0,
               min_detections: int = 3) -> DetectionResult:
        """
        Run the complete detection pipeline on a set of images.
        
        Args:
            image_paths: List of paths to image files
            detection_threshold: Detection threshold in sigma
            min_detections: Minimum detections required for confirmation
            
        Returns:
            DetectionResult with all detected objects
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        logger.info(f"Processing {len(image_paths)} images")
        
        # Step 1: Load and preprocess images
        total_steps = len(image_paths) * 4 + 3  # Load, detect, align, motion, identify
        
        with progress_context(total_steps, "Processing", self.progress_mode) as progress:
            
            images = []
            metadata_list = []
            sources_list = []
            times = []
            
            # Load images
            for i, path in enumerate(image_paths):
                progress.update(0, f"Loading {path.name}")
                
                try:
                    data, meta = self.image_processor.load_image(path)
                    data = self.image_processor.preprocess(data)
                    images.append(data)
                    metadata_list.append(meta)
                    
                    if meta.observation_time:
                        times.append(meta.observation_time)
                    else:
                        # Estimate time from file modification
                        times.append(datetime.fromtimestamp(path.stat().st_mtime))
                    
                except Exception as e:
                    errors.append(f"Failed to load {path}: {e}")
                    logger.error(f"Failed to load {path}: {e}")
                
                progress.update(1)
            
            if len(images) < 2:
                raise ValueError("Need at least 2 valid images for detection")
            
            # Detect sources in each image
            for i, (image, meta) in enumerate(zip(images, metadata_list)):
                progress.update(0, f"Detecting sources in image {i+1}")
                
                try:
                    sources = self.star_detector.detect(image, detection_threshold)
                    
                    # Add celestial coordinates if WCS available
                    if meta.wcs is not None:
                        for src in sources:
                            coords = self.plate_solver.solve_from_wcs(
                                meta.wcs, [(src.x, src.y)]
                            )
                            if coords and coords[0][0] is not None:
                                src.ra, src.dec = coords[0]
                    
                    sources_list.append(sources)
                    
                except Exception as e:
                    errors.append(f"Source detection failed for image {i}: {e}")
                    sources_list.append([])
                
                progress.update(1)
            
            # Step 2: Detect motion
            progress.update(0, "Analyzing motion")
            
            try:
                self.motion_detector.min_detections = min_detections
                moving_objects = self.motion_detector.detect_motion(
                    images, sources_list, times
                )
            except Exception as e:
                errors.append(f"Motion detection failed: {e}")
                moving_objects = []
            
            progress.update(1)
            
            # Step 3: Add celestial coordinates to moving objects
            progress.update(0, "Computing coordinates")
            
            for obj in moving_objects:
                wcs = metadata_list[0].wcs
                if wcs:
                    coords = self.plate_solver.solve_from_wcs(wcs, obj.positions)
                    obj.ra_positions = [c[0] for c in coords if c[0] is not None]
                    obj.dec_positions = [c[1] for c in coords if c[1] is not None]
            
            progress.update(1)
            
            # Step 4: Identify objects
            progress.update(0, "Identifying objects")
            
            potential_discoveries = []
            known_objects = []
            
            for obj in moving_objects:
                identified = self.object_identifier.identify(obj)
                
                if identified.is_known:
                    known_objects.append(identified)
                else:
                    potential_discoveries.append(identified)
            
            progress.update(1)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Detection complete in {processing_time:.1f}s")
        logger.info(f"Found {len(moving_objects)} moving objects")
        logger.info(f"  - Known objects: {len(known_objects)}")
        logger.info(f"  - Potential discoveries: {len(potential_discoveries)}")
        
        return DetectionResult(
            input_files=[str(p) for p in image_paths],
            processing_time=processing_time,
            sources_per_image=[len(s) for s in sources_list],
            moving_objects=moving_objects,
            potential_discoveries=potential_discoveries,
            known_objects=known_objects,
            errors=errors,
            warnings=warnings
        )
    
    def generate_report(self, result: DetectionResult, 
                        output_path: Optional[Path] = None) -> str:
        """Generate a markdown report from detection results."""
        lines = [
            "# Asteroid Detection Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Processing Time:** {result.processing_time:.2f} seconds",
            "",
            "## Input Files",
            ""
        ]
        
        for f in result.input_files:
            lines.append(f"- {f}")
        
        lines.extend([
            "",
            "## Detection Summary",
            "",
            f"- **Total Moving Objects:** {len(result.moving_objects)}",
            f"- **Known Objects:** {len(result.known_objects)}",
            f"- **Potential Discoveries:** {len(result.potential_discoveries)}",
            ""
        ])
        
        if result.potential_discoveries:
            lines.extend([
                "## Potential Discoveries",
                "",
                "| ID | Velocity (arcsec/hr) | Position Angle | Confidence |",
                "|:---|:--------------------:|:--------------:|:----------:|"
            ])
            
            for obj in result.potential_discoveries:
                lines.append(
                    f"| {obj.id} | {obj.velocity_arcsec_per_hour:.2f} | "
                    f"{obj.position_angle:.1f}° | {obj.confidence:.2%} |"
                )
            
            lines.append("")
        
        if result.known_objects:
            lines.extend([
                "## Identified Known Objects",
                "",
                "| ID | Name/Designation | Velocity (arcsec/hr) |",
                "|:---|:-----------------|:--------------------:|"
            ])
            
            for obj in result.known_objects:
                name = obj.matched_name or obj.matched_designation or "Unknown"
                lines.append(
                    f"| {obj.id} | {name} | {obj.velocity_arcsec_per_hour:.2f} |"
                )
            
            lines.append("")
        
        if result.errors:
            lines.extend([
                "## Errors",
                ""
            ])
            for err in result.errors:
                lines.append(f"- {err}")
            lines.append("")
        
        if result.warnings:
            lines.extend([
                "## Warnings",
                ""
            ])
            for warn in result.warnings:
                lines.append(f"- {warn}")
        
        report = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for asteroid detection."""
    parser = argparse.ArgumentParser(
        description="Detect moving objects (asteroids, comets) in astronomical images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection with progress bar
  python asteroid_detector.py image1.fits image2.fits image3.fits

  # Detection with debug output
  python asteroid_detector.py --debug -v *.fits

  # Output results to JSON
  python asteroid_detector.py --output results.json --format json *.fits

  # Generate markdown report
  python asteroid_detector.py --output report.md --format markdown *.fits

Supported formats: FITS, TIFF, JPEG, XISF
        """
    )
    
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Input image files (at least 2 required)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file for results (default: stdout)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["json", "markdown", "text"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=3.0,
        help="Detection threshold in sigma (default: 3.0)"
    )
    
    parser.add_argument(
        "-m", "--min-detections",
        type=int,
        default=3,
        help="Minimum detections to confirm object (default: 3)"
    )
    
    parser.add_argument(
        "--astrometry-key",
        type=str,
        help="API key for astrometry.net plate solving"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write logs to file"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers
    )
    
    # Validate inputs
    if len(args.images) < 2:
        parser.error("At least 2 images are required for motion detection")
    
    for img in args.images:
        if not img.exists():
            parser.error(f"Image file not found: {img}")
    
    # Initialize detector
    progress_mode = "none" if args.no_progress else "tqdm"
    
    detector = AsteroidDetector(
        debug=args.debug,
        verbose=args.verbose,
        progress_mode=progress_mode,
        astrometry_api_key=args.astrometry_key
    )
    
    # Run detection
    try:
        result = detector.detect(
            args.images,
            detection_threshold=args.threshold,
            min_detections=args.min_detections
        )
        
        # Format output
        if args.format == "json":
            output = json.dumps({
                'input_files': result.input_files,
                'processing_time': result.processing_time,
                'sources_per_image': result.sources_per_image,
                'moving_objects': [
                    {
                        'id': obj.id,
                        'velocity_arcsec_per_hour': obj.velocity_arcsec_per_hour,
                        'position_angle': obj.position_angle,
                        'confidence': obj.confidence,
                        'is_known': obj.is_known,
                        'matched_name': obj.matched_name,
                        'positions': obj.positions,
                        'ra_positions': obj.ra_positions,
                        'dec_positions': obj.dec_positions
                    }
                    for obj in result.moving_objects
                ],
                'errors': result.errors,
                'warnings': result.warnings
            }, indent=2)
        
        elif args.format == "markdown":
            output = detector.generate_report(result)
        
        else:  # text
            lines = [
                f"Asteroid Detection Results",
                f"==========================",
                f"Processed {len(result.input_files)} images in {result.processing_time:.2f}s",
                f"",
                f"Moving Objects Found: {len(result.moving_objects)}",
                f"  - Known: {len(result.known_objects)}",
                f"  - Potential Discoveries: {len(result.potential_discoveries)}",
                ""
            ]
            
            for obj in result.moving_objects:
                status = "KNOWN" if obj.is_known else "POTENTIAL DISCOVERY"
                name = obj.matched_name or obj.matched_designation or "Unknown"
                lines.append(
                    f"  {obj.id}: {status} - {name if obj.is_known else ''}"
                    f" v={obj.velocity_arcsec_per_hour:.2f}\"/hr "
                    f"PA={obj.position_angle:.1f}° "
                    f"conf={obj.confidence:.0%}"
                )
            
            if result.errors:
                lines.extend(["", "Errors:"] + [f"  - {e}" for e in result.errors])
            
            output = "\n".join(lines)
        
        # Write output
        if args.output:
            args.output.write_text(output)
            print(f"Results written to {args.output}")
        else:
            print(output)
        
        # Exit with error code if there were errors
        sys.exit(1 if result.errors else 0)
        
    except Exception as e:
        logger.exception(f"Detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
