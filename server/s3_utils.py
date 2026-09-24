#!/usr/bin/env python3
"""
S3 Utilities for file download and upload operations

This module provides utilities for downloading files from S3 and uploading files to S3.

Requirements:
- boto3 library installed
- AWS credentials configured (via AWS CLI, environment variables, or IAM role)

Usage Examples:

1. Download file from S3:
    from s3_download import download_from_s3

    # Using S3 URL
    local_path = download_from_s3(
        s3_url="s3://bucket-name/folder/file.pdf",
        download_folder="local_folder"
    )

    # Using bucket and key separately
    local_path = download_from_s3(
        bucket_name="bucket-name",
        s3_key="folder/file.pdf",
        download_folder="local_folder"
    )

    # Using a presigned S3 https:// URL - downloaded via plain HTTP using the
    # signature already in the URL, no AWS credentials needed. Pass it as
    # given; do NOT run it through to_s3_uri() first (that strips the
    # signature and forces a credentialed boto3 download instead).
    local_path = download_from_s3(
        s3_url="https://bucket-name.s3.us-east-2.amazonaws.com/folder/file.pdf?X-Amz-...",
        download_folder="local_folder"
    )

2. Upload file to S3:
    from s3_download import upload_to_s3
    
    # Upload with original filename
    s3_url = upload_to_s3(
        local_file_path="local/path/file.pdf",
        s3_folder_uri="s3://bucket-name/upload-folder/"
    )
    
    # Upload with custom filename
    s3_url = upload_to_s3(
        local_file_path="local/path/file.pdf",
        s3_folder_uri="s3://bucket-name/upload-folder/",
        custom_filename="renamed_file.pdf"
    )
    
    # Upload with job_id to organize files in folders
    s3_url = upload_to_s3(
        local_file_path="local/path/file.pdf",
        s3_folder_uri="s3://bucket-name/upload-folder/",
        job_id="job_12345"
    )
    # Result: s3://bucket-name/upload-folder/job_12345/file.pdf
"""

import boto3
import logging
import os
import requests
import time
from datetime import datetime
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

S3_AWS_REGION = os.environ.get("S3_AWS_REGION", "us-east-2")


def to_s3_uri(url):
    """Turn an https S3 link (usually PRESIGNED) back into a plain s3://bucket/key URI.

    Why: the conversation listing hands images back as presigned https links that expire
    (X-Amz-Expires=3600). Feeding one of those into a later edit works for an hour and then
    starts 403-ing, and storing one keeps a dead link on the turn. The s3:// form never
    expires and our download path already handles it.

    Handles both layouts the backend produces:
        https://<bucket>.s3[.<region>].amazonaws.com/<key>?<signature>
        https://s3.<region>.amazonaws.com/<bucket>/<key>?<signature>
    Anything that isn't an S3 https link (including an s3:// URI already) comes back
    unchanged, so this is safe to call on any image url.
    """
    if not url or not isinstance(url, str):
        return url
    if url.startswith("s3://"):
        return url
    if not url.startswith("http"):
        return url

    try:
        parsed = urlparse(url)
        host = parsed.netloc
        key = unquote(parsed.path).lstrip("/")          # drops the ?X-Amz-... signature
        if "amazonaws.com" not in host or not key:
            return url

        if host.startswith("s3.") or host.startswith("s3-"):
            # path-style: the first path segment is the bucket
            bucket, _, key = key.partition("/")
        else:
            # virtual-hosted: the bucket is the leading label of the host
            bucket = host.split(".s3")[0]

        # The backend sometimes emits a bucket that still carries the endpoint suffix
        # (".../metabull-....s3.amazonaws.com/175/x.png"); strip it back to the real name.
        if ".s3" in bucket:
            bucket = bucket.split(".s3")[0]

        if not bucket or not key:
            return url
        return f"s3://{bucket}/{key}"
    except Exception:
        return url


def _is_s3_https_url(url):
    """True for an https(s) link that's actually hosted on S3 (same host check
    to_s3_uri already applies) - the download_from_s3 http(s) branch below
    stays scoped to real S3 links this way, not "fetch any URL", since
    download_from_s3 is one of the few internal functions a generated node is
    allowed to call (see node_validator.py's _ALLOWED_INTERNAL_NAMES) and that
    allowlist exists specifically to keep generated code from reaching an
    arbitrary host."""
    try:
        parsed = urlparse(url)
        return "amazonaws.com" in parsed.netloc and bool(unquote(parsed.path).lstrip("/"))
    except Exception:
        return False


def _download_via_presigned_url(url, download_folder):
    """Plain HTTP GET straight to `url`, signature and all - for a presigned
    S3 https:// link this needs no AWS credentials at all, since the
    signature in the query string IS the authorization (unlike the s3://
    branch below, which always needs this process's own real AWS
    credentials). Callers must NOT pre-convert a presigned URL through
    to_s3_uri() before reaching here - that strips the signature and forces
    an authenticated boto3 call this process may not be able to make."""
    os.makedirs(download_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    microseconds = int(time.time() * 1000000) % 1000000
    key = unquote(urlparse(url).path).lstrip("/")
    file_extension = f".{key.rsplit('.', 1)[-1]}" if "." in key else ""
    local_file_path = os.path.join(download_folder, f"{timestamp}_{microseconds}{file_extension}")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(local_file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_file_path
    except Exception as e:
        logger.warning(f"S3 presigned-URL download failed url={url}: {e}")
        return None


def download_from_s3(bucket_name=None, s3_key=None, download_folder=None, s3_url=None):
    """Download a file from S3 bucket to a folder with a random time-based filename

    Args:
        bucket_name (str): S3 bucket name
        s3_key (str): S3 object key
        download_folder (str): Local folder path to save the downloaded file
        s3_url (str): Either an s3://bucket-name/key URI (downloaded via boto3 -
            needs this process's own AWS credentials), or a presigned S3
            https:// URL (downloaded via a plain HTTP GET using the URL as
            given - needs no AWS credentials, since the signature already
            authorizes the request). Do not pre-convert a presigned URL
            through to_s3_uri() first - see _download_via_presigned_url.

    Returns:
        str: Full path to the downloaded file, or None if failed
    """
    if s3_url and s3_url.startswith(("http://", "https://")):
        if not download_folder or not _is_s3_https_url(s3_url):
            return None
        return _download_via_presigned_url(s3_url, download_folder)

    # Parse S3 URL if provided
    if s3_url:
        if not s3_url.startswith('s3://'):
            return None

        url_parts = s3_url[5:].split('/', 1)
        if len(url_parts) != 2:
            return None
        
        bucket_name = url_parts[0]
        s3_key = url_parts[1]
    
    # Validate required parameters
    if not bucket_name or not s3_key or not download_folder:
        return None
    
    # Create download folder if it doesn't exist
    os.makedirs(download_folder, exist_ok=True)
    
    # Generate random filename based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    microseconds = int(time.time() * 1000000) % 1000000
    
    # Get file extension from S3 key
    file_extension = ""
    if '.' in s3_key:
        file_extension = '.' + s3_key.split('.')[-1]
    
    # Create random filename
    random_filename = f"{timestamp}_{microseconds}{file_extension}"
    local_file_path = os.path.join(download_folder, random_filename)
    
    for region in ['us-east-1', 'us-east-2']:
        try:
            s3_client = boto3.client('s3', region_name=region)
            s3_client.download_file(bucket_name, s3_key, local_file_path)
            return local_file_path
        except Exception as e:
            logger.warning(f"S3 download failed region={region} bucket={bucket_name} key={s3_key}: {e}")
            continue
    return None

def upload_to_s3(local_file_path, s3_folder_uri, custom_filename=None, job_id=None):
    """Upload a file to S3 folder
    
    Args:
        local_file_path (str): Path to the local file to upload
        s3_folder_uri (str): S3 folder URI in format s3://bucket-name/folder/path/
        custom_filename (str, optional): Custom filename for S3. If None, uses original filename
        job_id (str, optional): Job ID to create a folder structure. Files will be stored in s3://bucket/folder/job_id/filename
    
    Returns:
        str: Full S3 URL of the uploaded file, or None if failed
    """
    
    # Validate local file exists
    if not os.path.exists(local_file_path):
        logger.error(f"Local file does not exist: {local_file_path}")
        return None

    # Parse S3 folder URI
    if not s3_folder_uri.startswith('s3://'):
        logger.error(f"Invalid S3 URI format: {s3_folder_uri}")
        return None
    
    # Remove s3:// prefix and split bucket/folder
    uri_parts = s3_folder_uri[5:].rstrip('/')
    if '/' in uri_parts:
        bucket_name = uri_parts.split('/')[0]
        s3_folder = '/'.join(uri_parts.split('/')[1:])
    else:
        bucket_name = uri_parts
        s3_folder = ""
    
    # Determine filename
    if custom_filename:
        filename = custom_filename
    else:
        filename = os.path.basename(local_file_path)
    
    # Build S3 key with job_id folder if provided
    path_parts = []
    if s3_folder:
        path_parts.append(s3_folder)
    if job_id:
        path_parts.append(job_id)
    path_parts.append(filename)

    # Coerce every part to str: a caller may pass an int id (e.g. user_id), and join on a
    # non-str raises "expected str instance, int found".
    s3_key = '/'.join(str(p) for p in path_parts)
    
    try:
        # Create S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Upload file to S3
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        
        # Return full S3 URL
        s3_url = f"s3://{bucket_name}/{s3_key}"
        logger.info(f"File uploaded successfully to: {s3_url}")
        return s3_url

    except Exception as e:
        logger.error(f"Error uploading file to S3: {str(e)}")
        return None


def generate_presigned_url(s3_url, expires_in=3600):
    """Turn a plain s3://bucket/key URI into a presigned, publicly-downloadable https URL.

    upload_to_s3() only ever returns bare s3:// URIs. Those are fine as long as everything
    downstream reads them with this process's own AWS credentials (e.g. image_gen_utils.
    download_image already special-cases s3:// via boto3). But an external generation
    provider (Kling, Runway, Veo, ...) fetches an image_url over the public internet with
    no AWS credentials of its own, so an s3:// URI handed to one of those silently fails.

    Anything that isn't an s3:// URI (already presigned / public https) is returned
    unchanged, so this is safe to call on any image url.
    """
    if not s3_url or not isinstance(s3_url, str) or not s3_url.startswith("s3://"):
        return s3_url

    try:
        bucket_name, s3_key = s3_url[len("s3://"):].split("/", 1)
    except ValueError:
        logger.error(f"Malformed s3:// URI, expected s3://bucket/key: {s3_url}")
        return s3_url

    try:
        s3_client = boto3.client("s3", region_name=S3_AWS_REGION)
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": s3_key},
            ExpiresIn=expires_in,
        )
        logger.info(f"Generated presigned URL for {s3_url} region={S3_AWS_REGION} (expires_in={expires_in}s) : {presigned_url}")
        return presigned_url
    except Exception as e:
        logger.error(f"Could not generate presigned URL for {s3_url}: ERROR : {e}")
        return s3_url

if __name__ == "__main__":
    # Configuration
    DOWNLOAD_FOLDER = "trige_files"
    
    S3_URL = "s3://test-shot-create/sample1.json"
    
    # Download file
    downloaded_path = download_from_s3(s3_url=S3_URL, download_folder=DOWNLOAD_FOLDER)
    
    if downloaded_path:
        print(f"Downloaded file to: {downloaded_path}")
    else:
        print("Failed to download file") 
    '''
    upload_this_file = "/Users/vaishaparam/Documents/Work/MetaBall/GitHub/MCP-Blender-SceneIO/test_upload.txt"
    # Test upload function
    S3_UPLOAD_FOLDER = "s3://metabull-ai-accelerator-triage-scene2shot"
    uploaded_url = upload_to_s3(upload_this_file, S3_UPLOAD_FOLDER)
    
    if uploaded_url:
        print(f"Uploaded file to: {uploaded_url}")
    else:
        print("Failed to upload file")
    '''