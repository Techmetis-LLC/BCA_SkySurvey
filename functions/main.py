#!/usr/bin/env python3
"""
Asteroid Detection Cloud Module
================================
Cloud-optimized version for GCP Cloud Functions deployment.
"""

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from google.cloud import storage, firestore
import functions_framework
from flask import Request, jsonify
from flask_cors import cross_origin

# Import the core detector
from asteroid_detector import (
    AsteroidDetector,
    ImageProcessor,
    DetectionResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
storage_client = storage.Client()
firestore_client = firestore.Client()

# Configuration
BUCKET_NAME = os.environ.get('STORAGE_BUCKET', 'asteroid-detection-uploads')
RESULTS_COLLECTION = 'detection_results'
JOBS_COLLECTION = 'detection_jobs'


def verify_firebase_token(request: Request) -> Optional[Dict]:
    """Verify Firebase authentication token."""
    auth_header = request.headers.get('Authorization', '')
    
    if not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split('Bearer ')[1]
    
    try:
        import firebase_admin
        from firebase_admin import auth
        
        # Initialize Firebase Admin if not already done
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return None


@functions_framework.http
@cross_origin()
def create_detection_job(request: Request):
    """
    Create a new asteroid detection job.
    
    Request body:
    {
        "image_urls": ["gs://bucket/image1.fits", ...],
        "options": {
            "threshold": 3.0,
            "min_detections": 3
        }
    }
    """
    # Verify authentication
    user = verify_firebase_token(request)
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        image_urls = data.get('image_urls', [])
        options = data.get('options', {})
        
        if len(image_urls) < 2:
            return jsonify({'error': 'At least 2 images required'}), 400
        
        # Create job document
        job_id = str(uuid.uuid4())
        job_doc = {
            'job_id': job_id,
            'user_id': user['uid'],
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'image_urls': image_urls,
            'options': options,
            'progress': 0,
            'result': None,
            'error': None
        }
        
        # Save to Firestore
        firestore_client.collection(JOBS_COLLECTION).document(job_id).set(job_doc)
        
        logger.info(f"Created job {job_id} for user {user['uid']}")
        
        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Detection job created successfully'
        }), 201
        
    except Exception as e:
        logger.exception(f"Failed to create job: {e}")
        return jsonify({'error': str(e)}), 500


@functions_framework.http
@cross_origin()
def process_detection_job(request: Request):
    """
    Process an asteroid detection job.
    Called by Cloud Tasks or directly for synchronous processing.
    
    Request body:
    {
        "job_id": "uuid"
    }
    """
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        
        if not job_id:
            return jsonify({'error': 'job_id required'}), 400
        
        # Get job from Firestore
        job_ref = firestore_client.collection(JOBS_COLLECTION).document(job_id)
        job_doc = job_ref.get()
        
        if not job_doc.exists:
            return jsonify({'error': 'Job not found'}), 404
        
        job_data = job_doc.to_dict()
        
        if job_data['status'] not in ['pending', 'retry']:
            return jsonify({'error': f"Invalid job status: {job_data['status']}"}), 400
        
        # Update status to processing
        job_ref.update({
            'status': 'processing',
            'updated_at': datetime.utcnow()
        })
        
        # Download images to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            local_files = []
            
            for i, url in enumerate(job_data['image_urls']):
                # Parse GCS URL
                if url.startswith('gs://'):
                    parts = url[5:].split('/', 1)
                    bucket_name = parts[0]
                    blob_name = parts[1]
                    
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    
                    # Determine file extension
                    ext = Path(blob_name).suffix or '.fits'
                    local_file = temp_path / f"image_{i:03d}{ext}"
                    
                    blob.download_to_filename(str(local_file))
                    local_files.append(local_file)
                    
                    # Update progress
                    progress = int((i + 1) / len(job_data['image_urls']) * 30)
                    job_ref.update({'progress': progress})
            
            # Run detection
            options = job_data.get('options', {})
            detector = AsteroidDetector(
                debug=False,
                verbose=True,
                progress_mode='none'
            )
            
            # Custom progress callback
            def update_progress(current, total, status):
                progress = 30 + int((current / total) * 60)
                job_ref.update({
                    'progress': progress,
                    'status_message': status
                })
            
            result = detector.detect(
                local_files,
                detection_threshold=options.get('threshold', 3.0),
                min_detections=options.get('min_detections', 3)
            )
            
            # Serialize result
            result_dict = {
                'input_files': [str(f.name) for f in local_files],
                'processing_time': result.processing_time,
                'sources_per_image': result.sources_per_image,
                'moving_objects_count': len(result.moving_objects),
                'known_objects_count': len(result.known_objects),
                'potential_discoveries_count': len(result.potential_discoveries),
                'moving_objects': [
                    {
                        'id': obj.id,
                        'velocity_arcsec_per_hour': obj.velocity_arcsec_per_hour,
                        'position_angle': obj.position_angle,
                        'confidence': obj.confidence,
                        'is_known': obj.is_known,
                        'matched_name': obj.matched_name,
                        'matched_designation': obj.matched_designation,
                        'positions': obj.positions,
                        'ra_positions': obj.ra_positions,
                        'dec_positions': obj.dec_positions
                    }
                    for obj in result.moving_objects
                ],
                'errors': result.errors,
                'warnings': result.warnings
            }
            
            # Generate markdown report
            report = detector.generate_report(result)
            
            # Upload report to storage
            report_blob_name = f"results/{job_id}/report.md"
            bucket = storage_client.bucket(BUCKET_NAME)
            report_blob = bucket.blob(report_blob_name)
            report_blob.upload_from_string(report, content_type='text/markdown')
            
            result_dict['report_url'] = f"gs://{BUCKET_NAME}/{report_blob_name}"
            
            # Update job with results
            job_ref.update({
                'status': 'completed',
                'progress': 100,
                'updated_at': datetime.utcnow(),
                'completed_at': datetime.utcnow(),
                'result': result_dict
            })
            
            logger.info(f"Job {job_id} completed successfully")
            
            return jsonify({
                'job_id': job_id,
                'status': 'completed',
                'result': result_dict
            }), 200
            
    except Exception as e:
        logger.exception(f"Job processing failed: {e}")
        
        # Update job with error
        if job_id:
            job_ref.update({
                'status': 'failed',
                'updated_at': datetime.utcnow(),
                'error': str(e)
            })
        
        return jsonify({'error': str(e)}), 500


@functions_framework.http
@cross_origin()
def get_job_status(request: Request):
    """
    Get the status of a detection job.
    
    Query params:
        job_id: The job ID to check
    """
    user = verify_firebase_token(request)
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    job_id = request.args.get('job_id')
    
    if not job_id:
        return jsonify({'error': 'job_id required'}), 400
    
    try:
        job_doc = firestore_client.collection(JOBS_COLLECTION).document(job_id).get()
        
        if not job_doc.exists:
            return jsonify({'error': 'Job not found'}), 404
        
        job_data = job_doc.to_dict()
        
        # Verify user owns this job
        if job_data['user_id'] != user['uid']:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Convert datetime objects for JSON serialization
        response_data = {
            'job_id': job_data['job_id'],
            'status': job_data['status'],
            'progress': job_data.get('progress', 0),
            'created_at': job_data['created_at'].isoformat() if job_data.get('created_at') else None,
            'updated_at': job_data['updated_at'].isoformat() if job_data.get('updated_at') else None,
            'completed_at': job_data['completed_at'].isoformat() if job_data.get('completed_at') else None,
            'error': job_data.get('error'),
            'result': job_data.get('result')
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.exception(f"Failed to get job status: {e}")
        return jsonify({'error': str(e)}), 500


@functions_framework.http
@cross_origin()
def get_signed_upload_url(request: Request):
    """
    Generate a signed URL for uploading images.
    
    Request body:
    {
        "filename": "image.fits",
        "content_type": "application/fits"
    }
    """
    user = verify_firebase_token(request)
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        content_type = data.get('content_type', 'application/octet-stream')
        
        if not filename:
            return jsonify({'error': 'filename required'}), 400
        
        # Validate file extension
        ext = Path(filename).suffix.lower()
        valid_extensions = {'.fits', '.fit', '.fts', '.tiff', '.tif', '.jpg', '.jpeg', '.xisf'}
        
        if ext not in valid_extensions:
            return jsonify({
                'error': f'Invalid file type. Supported: {", ".join(valid_extensions)}'
            }), 400
        
        # Generate unique blob name
        upload_id = str(uuid.uuid4())
        blob_name = f"uploads/{user['uid']}/{upload_id}/{filename}"
        
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        
        # Generate signed URL (valid for 1 hour)
        url = blob.generate_signed_url(
            version='v4',
            expiration=3600,
            method='PUT',
            content_type=content_type
        )
        
        return jsonify({
            'upload_url': url,
            'blob_url': f"gs://{BUCKET_NAME}/{blob_name}",
            'expires_in': 3600
        }), 200
        
    except Exception as e:
        logger.exception(f"Failed to generate upload URL: {e}")
        return jsonify({'error': str(e)}), 500


@functions_framework.http
@cross_origin()
def list_user_jobs(request: Request):
    """
    List all detection jobs for the authenticated user.
    
    Query params:
        limit: Maximum number of jobs to return (default: 20)
        status: Filter by status (optional)
    """
    user = verify_firebase_token(request)
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        limit = int(request.args.get('limit', 20))
        status_filter = request.args.get('status')
        
        query = firestore_client.collection(JOBS_COLLECTION)\
            .where('user_id', '==', user['uid'])\
            .order_by('created_at', direction=firestore.Query.DESCENDING)\
            .limit(limit)
        
        if status_filter:
            query = query.where('status', '==', status_filter)
        
        jobs = []
        for doc in query.stream():
            job_data = doc.to_dict()
            jobs.append({
                'job_id': job_data['job_id'],
                'status': job_data['status'],
                'progress': job_data.get('progress', 0),
                'created_at': job_data['created_at'].isoformat() if job_data.get('created_at') else None,
                'image_count': len(job_data.get('image_urls', [])),
                'moving_objects_count': job_data.get('result', {}).get('moving_objects_count', 0) if job_data.get('result') else None
            })
        
        return jsonify({'jobs': jobs}), 200
        
    except Exception as e:
        logger.exception(f"Failed to list jobs: {e}")
        return jsonify({'error': str(e)}), 500


@functions_framework.http
@cross_origin()  
def health_check(request: Request):
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200
