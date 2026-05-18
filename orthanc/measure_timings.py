#!/usr/bin/env python3
"""
Comprehensive timing profiler for AI routing pipeline

Measures end-to-end and component-level timings for:
1. orthanc-viewer → orthanc-router (DICOM send)
2. orthanc-router → ML model (HTTP request)
3. ML model processing (download, convert, inference)
4. orthanc-router → DICOM wrapping (SR/SC creation)
5. orthanc-router → orthanc-viewer (result send)

Usage:
    python measure_timings.py --study-id <orthanc-study-id> --target <ai-router-name>
    python measure_timings.py --series-uid <series-uid> --target orthanc-router
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TimingProfiler:
    """Tracks timing measurements for the AI routing pipeline"""

    def __init__(self, trace_id: str, output_dir: str = "./profiling_results"):
        self.trace_id = trace_id
        self.output_dir = output_dir
        self.measurements: List[Dict] = []
        self.start_time = time.time()
        self.start_datetime = datetime.now()  # For log filtering

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # CSV file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"timing_profile_{timestamp}_{trace_id[:8]}.csv")

        logger.info("="*80)
        logger.info(f"Profiling Session Started")
        logger.info(f"Trace ID: {trace_id}")
        logger.info(f"Output: {self.csv_path}")
        logger.info("="*80)

    def record(self, component: str, operation: str, duration_ms: float,
               metadata: Optional[Dict] = None):
        """Record a timing measurement"""
        measurement = {
            'trace_id': self.trace_id,
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'operation': operation,
            'duration_ms': round(duration_ms, 2),
            'metadata': json.dumps(metadata or {})
        }
        self.measurements.append(measurement)

        # Log to stdout
        logger.info(
            f"[{component}] {operation}: {duration_ms:.2f}ms"
            + (f" | {metadata}" if metadata else "")
        )

    def save_results(self):
        """Save all measurements to CSV"""
        if not self.measurements:
            logger.warning("No measurements to save")
            return

        # Write to CSV
        fieldnames = ['trace_id', 'timestamp', 'component', 'operation',
                      'duration_ms', 'metadata']

        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.measurements)

        # Print summary
        total_duration = (time.time() - self.start_time) * 1000
        self._print_summary(total_duration)

    def _print_summary(self, total_duration: float):
        """Print a summary of all timings"""
        logger.info("="*80)
        logger.info("PROFILING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total End-to-End Duration: {total_duration:.2f}ms ({total_duration/1000:.2f}s)")
        logger.info("")

        # Group by component
        by_component = {}
        for m in self.measurements:
            comp = m['component']
            if comp not in by_component:
                by_component[comp] = []
            by_component[comp].append(m)

        # Print breakdown by component
        for component, measurements in sorted(by_component.items()):
            component_total = sum(m['duration_ms'] for m in measurements)
            percentage = (component_total / total_duration * 100) if total_duration > 0 else 0

            logger.info(f"\n{component}: {component_total:.2f}ms ({percentage:.1f}%)")
            logger.info("-" * 60)

            for m in measurements:
                logger.info(f"  • {m['operation']:<40} {m['duration_ms']:>10.2f}ms")

        logger.info("")
        logger.info("="*80)
        logger.info(f"Results saved to: {self.csv_path}")
        logger.info("="*80)


def create_http_session() -> requests.Session:
    """Create HTTP session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_study_info(orthanc_url: str, study_id: str) -> Dict:
    """Get study information from Orthanc

    Args:
        orthanc_url: Base URL of Orthanc instance
        study_id: Either Orthanc internal ID or DICOM StudyInstanceUID

    Returns:
        Study information dictionary
    """
    # If study_id looks like a DICOM UID (contains dots), look it up first
    if '.' in study_id:
        logger.info(f"Study ID looks like DICOM UID, looking up Orthanc ID...")
        try:
            lookup_response = requests.post(
                f"{orthanc_url}/tools/lookup",
                data=study_id,
                headers={"Content-Type": "text/plain"}
            )
            lookup_response.raise_for_status()
            lookup_results = lookup_response.json()

            # Find the study result
            study_results = [r for r in lookup_results if r.get('Type') == 'Study']

            if not study_results:
                raise ValueError(f"StudyInstanceUID not found in Orthanc: {study_id}")

            orthanc_study_id = study_results[0]['ID']
            logger.info(f"Found Orthanc study ID: {orthanc_study_id}")
            study_id = orthanc_study_id

        except requests.exceptions.HTTPException as e:
            raise ValueError(f"Failed to lookup StudyInstanceUID: {e}")

    response = requests.get(f"{orthanc_url}/studies/{study_id}")
    response.raise_for_status()
    return response.json()


def get_series_info(orthanc_url: str, study_id: str) -> List[Dict]:
    """Get all series in a study"""
    study_info = get_study_info(orthanc_url, study_id)
    series_list = []

    for series_id in study_info.get('Series', []):
        series_response = requests.get(f"{orthanc_url}/series/{series_id}")
        series_response.raise_for_status()
        series_data = series_response.json()
        series_list.append({
            'id': series_id,
            'uid': series_data['MainDicomTags']['SeriesInstanceUID'],
            'description': series_data['MainDicomTags'].get('SeriesDescription', 'N/A'),
            'modality': series_data['MainDicomTags'].get('Modality', 'N/A'),
            'instances': len(series_data['Instances'])
        })

    return series_list


def delete_study_from_router(
    orthanc_router_url: str,
    study_instance_uid: str
) -> bool:
    """
    Delete study from target router to avoid overwrite overhead

    Args:
        orthanc_router_url: Base URL of the target Orthanc router
        study_instance_uid: DICOM StudyInstanceUID to delete

    Returns:
        True if deleted or didn't exist, False on error
    """
    try:
        # Look up the study in the router
        logger.info(f"Looking up study in router: {study_instance_uid}")
        lookup_response = requests.post(
            f"{orthanc_router_url}/tools/lookup",
            data=study_instance_uid,
            headers={"Content-Type": "text/plain"}
        )
        lookup_response.raise_for_status()
        lookup_results = lookup_response.json()

        # Find the study
        study_results = [r for r in lookup_results if r.get('Type') == 'Study']

        if not study_results:
            logger.info(f"Study not found in router (clean slate)")
            return True

        orthanc_study_id = study_results[0]['ID']
        logger.info(f"Found study in router: {orthanc_study_id}, deleting...")

        # Delete the study
        delete_start = time.time()
        delete_response = requests.delete(f"{orthanc_router_url}/studies/{orthanc_study_id}")
        delete_response.raise_for_status()
        delete_duration = (time.time() - delete_start) * 1000

        logger.info(f"Study deleted successfully ({delete_duration:.2f}ms)")
        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.info(f"Study not found in router (clean slate)")
            return True
        logger.error(f"Failed to delete study from router: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during study deletion: {e}")
        return False


def send_to_ai_and_profile(
    orthanc_viewer_url: str,
    study_id: str,
    target: str,
    target_url: str,
    profiler: TimingProfiler,
    series_uids: Optional[List[str]] = None
) -> Dict:
    """
    Send study to AI router and profile the operation

    Returns: Response from send-to-ai endpoint
    """
    session = create_http_session()

    # Prepare request
    payload = {
        "study_id": study_id,
        "target": target,
        "target_url": target_url
    }

    if series_uids:
        payload["series_uids"] = series_uids

    # Send request with timing
    logger.info(f"Sending study {study_id} to {target}...")
    start = time.time()

    try:
        # Use the same endpoint as frontend
        response = session.post(
            f"{orthanc_viewer_url}/send-to-ai",
            json=payload,
            timeout=300  # 5 minutes timeout
        )

        duration_ms = (time.time() - start) * 1000
        profiler.record(
            component="orthanc-viewer",
            operation="send_to_ai (total)",
            duration_ms=duration_ms,
            metadata={
                "study_id": study_id,
                "target": target,
                "status_code": response.status_code
            }
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        profiler.record(
            component="orthanc-viewer",
            operation="send_to_ai (FAILED)",
            duration_ms=duration_ms,
            metadata={"error": str(e)}
        )
        raise


def wait_for_ai_results(
    orthanc_viewer_url: str,
    study_id: str,
    profiler: TimingProfiler,
    initial_series_count: int,
    timeout: int = 300
) -> bool:
    """
    Poll orthanc-viewer for new AI result series

    Returns: True if new series detected, False if timeout
    """
    logger.info(f"Waiting for AI results (timeout: {timeout}s)...")
    start = time.time()
    poll_count = 0

    while (time.time() - start) < timeout:
        poll_count += 1
        poll_start = time.time()

        try:
            study_info = get_study_info(orthanc_viewer_url, study_id)
            current_series_count = len(study_info.get('Series', []))

            poll_duration = (time.time() - poll_start) * 1000

            if current_series_count > initial_series_count:
                total_wait = (time.time() - start) * 1000
                new_series = current_series_count - initial_series_count

                profiler.record(
                    component="orthanc-viewer",
                    operation="wait_for_ai_results",
                    duration_ms=total_wait,
                    metadata={
                        "polls": poll_count,
                        "new_series": new_series,
                        "initial_count": initial_series_count,
                        "final_count": current_series_count
                    }
                )

                logger.info(f"AI results received! {new_series} new series detected")
                return True

            # Wait before next poll
            time.sleep(2)

        except Exception as e:
            logger.warning(f"Error polling for results: {e}")
            time.sleep(2)

    # Timeout
    total_wait = (time.time() - start) * 1000
    profiler.record(
        component="orthanc-viewer",
        operation="wait_for_ai_results (TIMEOUT)",
        duration_ms=total_wait,
        metadata={"polls": poll_count, "timeout": timeout}
    )

    logger.error(f"Timeout waiting for AI results after {timeout}s")
    return False


def fetch_component_logs(profiler: TimingProfiler, containers: List[str]):
    """
    Fetch and parse timing logs from Docker containers

    This extracts structured timing information from service logs
    """
    logger.info("\nFetching component logs for timing analysis...")

    for container in containers:
        try:
            # Get logs from docker container with timestamps (use sudo if needed)
            import subprocess
            import os

            # Check if we need sudo
            # Use --timestamps to get timestamps for filtering
            docker_cmd = ['docker', 'logs', '--timestamps', '--since', '10m', container]
            if os.geteuid() != 0:  # Not running as root
                docker_cmd = ['sudo'] + docker_cmd

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Docker logs can be in either stdout or stderr, combine both
                logs = result.stdout + result.stderr
                parse_timing_logs(profiler, container, logs)
            else:
                logger.warning(f"Failed to get logs from {container}: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout getting logs from {container}")
        except FileNotFoundError:
            logger.warning("Docker command not found. Skipping log extraction.")
            break
        except Exception as e:
            logger.warning(f"Error getting logs from {container}: {e}")


def parse_timing_logs(profiler: TimingProfiler, container: str, logs: str):
    """
    Parse timing information from container logs

    Looks for patterns like:
    - "TIMING: operation_name: 123.45ms"
    - "Step X took Y seconds"
    - etc.

    Filters logs to only include entries after profiler start time
    """
    from datetime import datetime
    import re

    component_map = {
        'odelia-orthanc-viewer': 'orthanc-viewer',
        'odelia-orthanc-router': 'orthanc-router',
        'odelia-orthanc-router-mst': 'orthanc-router-mst',
        'odelia-orthanc-router-medgemma': 'orthanc-router-medgemma',
        'odelia-breast-cancer-classification': 'ml-breast-cancer',
        'odelia-mst-classifier': 'ml-mst',
        'odelia-medgemma-mri': 'ml-medgemma'
    }

    component = component_map.get(container, container)

    for line in logs.split('\n'):
        # Check if line has Docker timestamp (format: 2024-11-12T13:56:54.123456789Z)
        timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(.*)$', line)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            line_content = timestamp_match.group(2)

            # Parse timestamp and filter by profiler start time
            try:
                # Parse Docker timestamp (ISO format with Z)
                log_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                # Skip if log entry is before profiler started
                # Add timezone-naive comparison
                log_time_naive = log_time.replace(tzinfo=None)
                if log_time_naive < profiler.start_datetime:
                    continue

            except Exception as e:
                # If timestamp parsing fails, include the line anyway
                logger.debug(f"Could not parse timestamp: {timestamp_str} - {e}")
                line_content = line
        else:
            # No timestamp found, use full line (might be continuation or non-timestamped)
            line_content = line
        # Look for TIMING markers in the line content (handle both direct and logger-formatted output)
        if 'TIMING:' in line_content or 'PROFILE:' in line_content:
            try:
                # Expected formats:
                # "TIMING: operation_name: 123.45ms"
                # "INFO:__main__:TIMING: operation_name: 123.45ms"

                # Split on TIMING: or PROFILE:
                marker = 'TIMING:' if 'TIMING:' in line_content else 'PROFILE:'
                parts = line_content.split(marker)
                if len(parts) >= 2:
                    info = parts[1].strip()

                    # Parse operation and duration
                    # Format: "operation_name: 123.45ms"
                    if ':' in info:
                        colon_parts = info.split(':', 1)
                        operation = colon_parts[0].strip()

                        # Extract duration
                        duration_str = colon_parts[1].strip()
                        if 'ms' in duration_str:
                            duration = float(duration_str.split('ms')[0].strip())
                        elif 's' in duration_str and 'ms' not in duration_str:
                            duration = float(duration_str.split('s')[0].strip()) * 1000
                        else:
                            continue

                        # Extract metadata if present
                        metadata = {}
                        if '[' in duration_str and ']' in duration_str:
                            meta_str = duration_str.split('[')[1].split(']')[0]
                            try:
                                metadata = json.loads('{' + meta_str + '}')
                            except:
                                metadata = {'raw': meta_str}

                        profiler.record(
                            component=component,
                            operation=operation,
                            duration_ms=duration,
                            metadata=metadata
                        )
            except Exception as e:
                logger.debug(f"Could not parse timing line: {line_content[:100]} - {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Profile AI routing pipeline timings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile a specific study (using Orthanc ID)
  python measure_timings.py --study-id abc123 --target orthanc-router

  # Profile using DICOM StudyInstanceUID
  python measure_timings.py --study-id 1.2.840.113619.2.55... --target orthanc-router

  # Profile with clean slate (no overwrite overhead)
  python measure_timings.py --study-id abc123 --target orthanc-router --delete-before-send

  # Profile with specific series
  python measure_timings.py --study-id abc123 --target orthanc-router \\
      --series-uids 1.2.3.4 1.2.3.5

  # Profile MST model
  python measure_timings.py --study-id abc123 --target orthanc-router-mst

  # Profile MedGemma model
  python measure_timings.py --study-id abc123 --target orthanc-router-medgemma

  # Compare deletion overhead (run both):
  python measure_timings.py --study-id abc123 --target orthanc-router --delete-before-send
  python measure_timings.py --study-id abc123 --target orthanc-router
        """
    )

    parser.add_argument(
        '--study-id',
        required=True,
        help='Orthanc study ID or DICOM StudyInstanceUID to process'
    )

    parser.add_argument(
        '--target',
        default='orthanc-router',
        help='Target AI router (orthanc-router, orthanc-router-mst, or orthanc-router-medgemma)'
    )

    parser.add_argument(
        '--orthanc-viewer-url',
        default='http://localhost:8000',
        help='Orthanc viewer URL (default: http://localhost:8000)'
    )

    parser.add_argument(
        '--target-url',
        help='Target DICOM URL (default: auto-detect from target)'
    )

    parser.add_argument(
        '--series-uids',
        nargs='+',
        help='Optional: Specific series UIDs to send'
    )

    parser.add_argument(
        '--output-dir',
        default='./profiling_results',
        help='Output directory for results (default: ./profiling_results)'
    )

    parser.add_argument(
        '--no-log-extraction',
        action='store_true',
        help='Skip extracting timing logs from Docker containers'
    )

    parser.add_argument(
        '--delete-before-send',
        action='store_true',
        help='Delete study from target router before sending (measures clean transfer without overwrite overhead)'
    )

    args = parser.parse_args()

    # Generate trace ID
    trace_id = str(uuid.uuid4())

    # Initialize profiler
    profiler = TimingProfiler(trace_id, args.output_dir)

    # Auto-detect target URL if not provided (use DICOMweb URLs like frontend)
    if not args.target_url:
        target_map = {
            'orthanc-router': 'http://orthanc-router:8042/dicom-web',
            'orthanc-router-mst': 'http://orthanc-router-mst:8042/dicom-web',
            'orthanc-router-medgemma': 'http://orthanc-router-medgemma:8042/dicom-web'
        }
        args.target_url = target_map.get(args.target, 'http://orthanc-router:8042/dicom-web')

    try:
        # Get initial study state
        logger.info(f"\nFetching study information...")
        study_info = get_study_info(args.orthanc_viewer_url, args.study_id)

        # Get the actual Orthanc ID (in case user provided StudyInstanceUID)
        orthanc_study_id = study_info.get('ID')
        initial_series_count = len(study_info.get('Series', []))

        logger.info(f"Study ID (Orthanc): {orthanc_study_id}")
        logger.info(f"Study UID (DICOM): {study_info.get('MainDicomTags', {}).get('StudyInstanceUID', 'N/A')}")
        logger.info(f"Patient: {study_info.get('PatientMainDicomTags', {}).get('PatientName', 'N/A')}")
        logger.info(f"Initial series count: {initial_series_count}")


        # Get series details
        series_list = get_series_info(args.orthanc_viewer_url, orthanc_study_id)
        logger.info(f"\nSeries in study:")
        for s in series_list:
            logger.info(f"  • {s['description']} ({s['modality']}) - {s['instances']} instances")

        # Delete from router if requested (to measure clean transfer without overwrite overhead)
        if args.delete_before_send:
            logger.info(f"\n{'='*80}")
            logger.info("STEP 0: Deleting study from target router (clean slate)")
            logger.info(f"{'='*80}")

            study_uid = study_info.get('MainDicomTags', {}).get('StudyInstanceUID')
            if not study_uid:
                logger.error("Cannot delete: StudyInstanceUID not found")
            else:
                # Map target to localhost URL (script runs on host, not in Docker)
                router_url_map = {
                    'orthanc-router': 'http://localhost:8042',
                    'orthanc-router-mst': 'http://localhost:8043',
                    'orthanc-router-medgemma': 'http://localhost:8044'
                }
                router_base_url = router_url_map.get(args.target, 'http://localhost:8042')
                logger.info(f"Using router URL: {router_base_url}")
                delete_study_from_router(router_base_url, study_uid)

        # Start end-to-end timing
        e2e_start = time.time()

        # Send to AI
        logger.info(f"\n{'='*80}")
        logger.info("STEP 1: Sending to AI Router")
        logger.info(f"{'='*80}")

        send_response = send_to_ai_and_profile(
            orthanc_viewer_url=args.orthanc_viewer_url,
            study_id=orthanc_study_id,  # Use the Orthanc internal ID
            target=args.target,
            target_url=args.target_url,
            profiler=profiler,
            series_uids=args.series_uids
        )

        logger.info(f"Send response: {json.dumps(send_response, indent=2)}")

        # Wait for AI results
        logger.info(f"\n{'='*80}")
        logger.info("STEP 2: Waiting for AI Results")
        logger.info(f"{'='*80}")

        results_received = wait_for_ai_results(
            orthanc_viewer_url=args.orthanc_viewer_url,
            study_id=orthanc_study_id,  # Use the Orthanc internal ID
            profiler=profiler,
            initial_series_count=initial_series_count,
            timeout=300
        )

        # Record end-to-end timing
        e2e_duration = (time.time() - e2e_start) * 1000
        profiler.record(
            component="end-to-end",
            operation="complete_pipeline",
            duration_ms=e2e_duration,
            metadata={
                "study_id": orthanc_study_id,
                "target": args.target,
                "success": results_received
            }
        )

        # Extract timing from component logs
        if not args.no_log_extraction:
            logger.info(f"\n{'='*80}")
            logger.info("STEP 3: Extracting Component Timings from Logs")
            logger.info(f"{'='*80}")

            containers = [
                'odelia-orthanc-viewer',
                'odelia-orthanc-router',
                'odelia-orthanc-router-mst',
                'odelia-orthanc-router-medgemma',
                'odelia-breast-cancer-classification',
                'odelia-mst-classifier',
                'odelia-medgemma-mri'
            ]

            fetch_component_logs(profiler, containers)

        # Get final series info
        if results_received:
            logger.info(f"\nFinal series in study:")
            final_series = get_series_info(args.orthanc_viewer_url, orthanc_study_id)
            for s in final_series:
                logger.info(f"  • {s['description']} ({s['modality']}) - {s['instances']} instances")

        # Save results
        profiler.save_results()

        return 0 if results_received else 1

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        profiler.save_results()
        return 130
    except Exception as e:
        logger.error(f"Error during profiling: {e}", exc_info=True)
        profiler.save_results()
        return 1


if __name__ == '__main__':
    sys.exit(main())
