#!/usr/bin/env python3
"""
Tests for Asteroid Detection System
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from asteroid_detector import (
    ImageProcessor,
    StarDetector,
    MotionDetector,
    ImageRegistrar,
    PlateSolver,
    ObjectIdentifier,
    AsteroidDetector,
    DetectedSource,
    MovingObject,
    ImageMetadata
)


class TestImageProcessor:
    """Tests for ImageProcessor class."""
    
    def test_supported_formats(self):
        """Test that all expected formats are supported."""
        processor = ImageProcessor()
        expected = {'.jpg', '.jpeg', '.tiff', '.tif', '.fits', '.fit', '.fts', '.xisf'}
        assert processor.SUPPORTED_FORMATS == expected
    
    def test_preprocess_removes_hot_pixels(self):
        """Test that hot pixel removal works."""
        processor = ImageProcessor()
        
        # Create image with hot pixel
        data = np.random.normal(100, 10, (100, 100)).astype(np.float32)
        data[50, 50] = 10000  # Hot pixel
        
        processed = processor.preprocess(data, remove_hot_pixels=True)
        
        # Hot pixel should be reduced
        assert processed[50, 50] < 1000


class TestStarDetector:
    """Tests for StarDetector class."""
    
    def test_detect_sources(self):
        """Test source detection on synthetic image."""
        detector = StarDetector()
        
        # Create synthetic image with stars
        data = np.random.normal(100, 10, (200, 200)).astype(np.float32)
        
        # Add synthetic stars (Gaussian profiles)
        for x, y in [(50, 50), (100, 100), (150, 150)]:
            xx, yy = np.meshgrid(np.arange(200), np.arange(200))
            star = 5000 * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 3**2))
            data += star.astype(np.float32)
        
        sources = detector.detect(data, threshold=3.0)
        
        assert len(sources) >= 2  # Should detect at least 2 of the 3 stars
        assert all(isinstance(s, DetectedSource) for s in sources)
    
    def test_filter_edge_sources(self):
        """Test that edge sources are filtered."""
        detector = StarDetector()
        
        sources = [
            DetectedSource(x=10, y=100, flux=1000, snr=50, fwhm=3, ellipticity=0.1),
            DetectedSource(x=100, y=10, flux=1000, snr=50, fwhm=3, ellipticity=0.1),
            DetectedSource(x=100, y=100, flux=1000, snr=50, fwhm=3, ellipticity=0.1),
        ]
        
        filtered = detector._filter_sources(sources, (200, 200))
        
        # Only center source should remain
        assert len(filtered) == 1
        assert filtered[0].x == 100 and filtered[0].y == 100


class TestImageRegistrar:
    """Tests for ImageRegistrar class."""
    
    def test_compute_shift(self):
        """Test shift computation between images."""
        registrar = ImageRegistrar()
        
        # Create reference image
        ref = np.zeros((100, 100), dtype=np.float32)
        ref[40:60, 40:60] = 1000
        
        # Create shifted image
        target = np.zeros((100, 100), dtype=np.float32)
        target[45:65, 42:62] = 1000  # Shifted by (5, 2)
        
        dx, dy = registrar.compute_shift(ref, target)
        
        assert abs(dx - 2) < 1  # X shift should be ~2
        assert abs(dy - 5) < 1  # Y shift should be ~5


class TestMotionDetector:
    """Tests for MotionDetector class."""
    
    def test_linear_motion_check(self):
        """Test linear motion detection."""
        detector = MotionDetector()
        
        # Linear positions
        linear_positions = [(0, 0), (10, 10), (20, 20), (30, 30)]
        assert detector._is_linear_motion(linear_positions)
        
        # Non-linear positions
        nonlinear_positions = [(0, 0), (10, 50), (20, 20), (30, 30)]
        assert not detector._is_linear_motion(nonlinear_positions, tolerance=2)
    
    def test_filter_false_positives(self):
        """Test that false positives are filtered."""
        detector = MotionDetector()
        
        now = datetime.now()
        
        objects = [
            # Good detection
            MovingObject(
                id="good", positions=[(0,0), (10,10), (20,20)],
                times=[now, now + timedelta(hours=1), now + timedelta(hours=2)],
                ra_positions=[], dec_positions=[],
                velocity_arcsec_per_hour=50, position_angle=45, confidence=0.9
            ),
            # Too fast (satellite)
            MovingObject(
                id="fast", positions=[(0,0), (100,100), (200,200)],
                times=[now, now + timedelta(hours=1), now + timedelta(hours=2)],
                ra_positions=[], dec_positions=[],
                velocity_arcsec_per_hour=2000, position_angle=45, confidence=0.9
            ),
            # Too slow (star)
            MovingObject(
                id="slow", positions=[(0,0), (0.1,0.1), (0.2,0.2)],
                times=[now, now + timedelta(hours=1), now + timedelta(hours=2)],
                ra_positions=[], dec_positions=[],
                velocity_arcsec_per_hour=0.1, position_angle=45, confidence=0.9
            ),
        ]
        
        filtered = detector._filter_false_positives(objects)
        
        assert len(filtered) == 1
        assert filtered[0].id == "good"


class TestAsteroidDetector:
    """Tests for main AsteroidDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = AsteroidDetector(debug=True, verbose=True)
        
        assert detector.debug is True
        assert detector.verbose is True
        assert detector.image_processor is not None
        assert detector.star_detector is not None
        assert detector.motion_detector is not None
    
    def test_generate_report(self):
        """Test report generation."""
        from asteroid_detector import DetectionResult
        
        detector = AsteroidDetector()
        
        result = DetectionResult(
            input_files=["image1.fits", "image2.fits"],
            processing_time=10.5,
            sources_per_image=[100, 95],
            moving_objects=[],
            potential_discoveries=[],
            known_objects=[],
            errors=[],
            warnings=[]
        )
        
        report = detector.generate_report(result)
        
        assert "Asteroid Detection Report" in report
        assert "10.5" in report
        assert "image1.fits" in report


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.skipif(True, reason="Requires test images")
    def test_full_pipeline(self):
        """Test full detection pipeline with real images."""
        # This test would require actual test images
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
