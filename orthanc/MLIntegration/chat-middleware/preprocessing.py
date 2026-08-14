"""
DICOM preprocessing with configurable slice strategies.
Reuses core DICOM handling from medgemma-mri/preprocessing.py.
"""

import base64
import io
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
from models import RegionOfInterest, SliceSelection, SliceStrategy
from PIL import Image
from runtime_config import PreprocessingParams
from shared.config import StorageConfig
from shared.dicom_storage import save_datasets_to_folder

# Import shared utilities
from shared.wado_retrieval import retrieve_via_wado_rs

logger = logging.getLogger(__name__)

# (0008,0018) SOPInstanceUID. SimpleITK spells DICOM tags with a pipe.
SOP_INSTANCE_UID_TAG = "0008|0018"


class SliceSelectionError(ValueError):
    """A message named slices that cannot be resolved in the retrieved series.

    Distinct from a retrieval failure: the series arrived fine, but what the
    client asked for is not in it. Surfaced to the user rather than silently
    substituted, because substituting means answering about different pixels than
    the panel says it sent.
    """


def effective_params(
    params: PreprocessingParams, selection: SliceSelection | None = None
) -> PreprocessingParams:
    """
    The preprocessing parameters actually in force for one series.

    A message may carry its own recipe, which then wins over the service's global
    runtime config. That is what lets the panel's snapshot describe what was used:
    the runtime config is shared and mutable, so a second browser changing it
    between compose and send would otherwise silently rewrite the first browser's
    request.

    Args:
        params: Runtime preprocessing parameters (the fallback)
        selection: Per-message slice selection, if the client sent one

    Returns:
        Parameters to preprocess with -- a copy, never the caller's object
    """
    if selection is not None and selection.has_recipe():
        return PreprocessingParams(
            num_slices=selection.num_slices,
            slice_strategy=selection.slice_strategy,
            central_percentage=(
                selection.central_percentage
                if selection.central_percentage is not None
                else params.central_percentage
            ),
        )
    # Copied so a concurrent config update cannot change the recipe out from
    # under a request that has already been keyed against it.
    return replace(params)


def recipe_signature(
    params: PreprocessingParams, selection: SliceSelection | None = None
) -> tuple[str, ...]:
    """
    Everything that affects the images produced for one series.

    Feeds `image_cache.make_cache_key`. Kept here, next to the code that consumes
    each input, so that adding a preprocessing input has one place to change --
    miss it and the cache starts serving pixels from a different recipe.

    Args:
        params: Preprocessing parameters in force (see `effective_params`)
        selection: Per-message slice selection, if the client sent one

    Returns:
        A stable tuple of strings describing the recipe
    """
    # The crop changes the pixels, so it changes the cache entry. Left out, the
    # same slices requested with and without a region would share one entry and
    # the second request would get the first one's framing.
    roi = selection.roi if selection is not None else None
    roi_part = (f"roi:{roi.x:.6f},{roi.y:.6f},{roi.width:.6f},{roi.height:.6f}",) if roi else ()

    if selection is not None and selection.sop_instance_uids:
        # The named instances ARE the recipe; the parameters play no part.
        # Order matters: the images are sent in this order.
        return ("instances", *selection.sop_instance_uids, *roi_part)

    resolved = effective_params(params, selection)
    return (
        "recipe",
        str(resolved.num_slices),
        resolved.slice_strategy.value,
        str(resolved.central_percentage),
        *roi_part,
    )


def read_dicom_volume_with_instances(dicom_folder: Path) -> tuple[sitk.Image, list[str]]:
    """
    Read a DICOM series as a 3D volume, plus the SOPInstanceUID of each Z-slice.

    The UID list is what makes a client's slice selection resolvable: it maps the
    viewer's notion of "slice 27" onto this volume's Z index without either side
    assuming the other sorts the series the same way.

    It is returned EMPTY when the files do not map one-to-one onto the volume --
    an enhanced/multi-frame instance becomes many Z-slices from a single file, so
    there is no per-slice UID to report. An empty list means "cannot address this
    series slice by slice", never "slice 0 for everything".

    Args:
        dicom_folder: Path to folder containing DICOM files

    Returns:
        (SimpleITK Image, per-Z-index SOPInstanceUIDs or [])
    """

    dicom_path = Path(dicom_folder)

    # Verify DICOM files exist
    dicom_files = list(dicom_path.glob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_folder}")

    logger.info(f"Found {len(dicom_files)} DICOM files")

    # For simplicity, use standard GDCM series reader
    # The medgemma version handles 4D temporal series, but for chat we'll
    # just use the standard approach
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))

    if not dicom_names:
        raise ValueError(f"No valid DICOM series found in {dicom_folder}")

    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    logger.info(f"Read DICOM series: size={image.GetSize()}, spacing={image.GetSpacing()}")

    return image, _slice_instance_uids(reader, image, len(dicom_names))


def read_dicom_volume(dicom_folder: Path) -> sitk.Image:
    """
    Read DICOM series as a 3D volume using SimpleITK.

    Thin wrapper over `read_dicom_volume_with_instances` for callers that do not
    need per-slice identity.

    Args:
        dicom_folder: Path to folder containing DICOM files

    Returns:
        SimpleITK Image object
    """
    volume, _ = read_dicom_volume_with_instances(dicom_folder)
    return volume


def _slice_instance_uids(
    reader: sitk.ImageSeriesReader, image: sitk.Image, file_count: int
) -> list[str]:
    """Per-Z-index SOPInstanceUIDs, or [] when they cannot be established."""
    size = image.GetSize()
    depth = size[2] if len(size) >= 3 else 1

    if file_count != depth:
        # One file per slice is the assumption that makes a per-slice UID
        # meaningful. Multi-frame instances break it.
        logger.info(
            f"Series has {file_count} files for {depth} slices; "
            "per-slice instance UIDs unavailable"
        )
        return []

    uids: list[str] = []
    for index in range(depth):
        try:
            raw = reader.GetMetaData(index, SOP_INSTANCE_UID_TAG)
        except RuntimeError:
            logger.warning(f"Slice {index} carries no SOPInstanceUID; cannot address by instance")
            return []
        # DICOM UIDs are padded to an even length with a NUL.
        uid = raw.strip().strip("\x00").strip()
        if not uid:
            logger.warning(f"Slice {index} has an empty SOPInstanceUID")
            return []
        uids.append(uid)

    if len(set(uids)) != len(uids):
        # Duplicates would make a UID ambiguous, and picking the first match
        # would quietly send the wrong slice.
        logger.warning("Series reports duplicate SOPInstanceUIDs; cannot address by instance")
        return []

    return uids


def resolve_selected_indices(slice_uids: list[str], requested_uids: list[str]) -> list[int]:
    """
    Map requested SOPInstanceUIDs onto Z indices of the reconstructed volume.

    Order is the caller's: the images are sent in the order requested, which is
    the order the panel displayed them in.

    Args:
        slice_uids: Per-Z-index SOPInstanceUIDs from `read_dicom_volume_with_instances`
        requested_uids: The instances the message asked for

    Returns:
        Z indices, one per requested UID

    Raises:
        SliceSelectionError: if any requested UID is not part of this series
    """
    index_of = {uid: index for index, uid in enumerate(slice_uids)}

    resolved: list[int] = []
    missing: list[str] = []
    for uid in requested_uids:
        index = index_of.get(uid)
        if index is None:
            missing.append(uid)
        else:
            resolved.append(index)

    if missing:
        # All-or-nothing. Dropping the unresolvable ones would send a different
        # set of slices than the message claims, which is exactly the failure the
        # per-message snapshot exists to make impossible.
        logger.error(
            f"{len(missing)} of {len(requested_uids)} requested slices are not in this series "
            f"(first missing: {missing[0]})"
        )
        raise SliceSelectionError(
            f"{len(missing)} of {len(requested_uids)} selected slices are not part of the "
            "retrieved series. Re-select the slice range and send again."
        )

    return resolved


def normalize_slice(slice_array: np.ndarray) -> np.ndarray:
    """
    Normalize a slice to 0-255 range for PIL Image conversion.
    Uses percentile-based windowing for robust normalization.

    Adapted from medgemma-mri/preprocessing.py.

    Args:
        slice_array: 2D numpy array

    Returns:
        Normalized uint8 array
    """
    # Use percentile-based windowing (robust to outliers)
    p_low, p_high = np.percentile(slice_array, [1, 99])

    if p_high - p_low < 1e-6:
        # Constant image
        return np.zeros_like(slice_array, dtype=np.uint8)

    # Clip and normalize to 0-255
    normalized = np.clip(slice_array, p_low, p_high)
    return ((normalized - p_low) / (p_high - p_low) * 255).astype(np.uint8)


def extract_slices(volume_array: np.ndarray, params: PreprocessingParams) -> list[np.ndarray]:
    """
    Extract slices based on the configured strategy.

    Strategies:
    - CENTRAL: Extract from central N% of volume (default, matches existing behavior)
    - UNIFORM: Evenly spaced across entire volume
    - FIRST_N: First N slices (useful for head-first scans)
    - LAST_N: Last N slices (useful for feet-first scans)

    Args:
        volume_array: 3D numpy array with shape (Z, Y, X)
        params: Preprocessing parameters including strategy

    Returns:
        List of 2D slice arrays
    """
    total_slices = volume_array.shape[0]
    num_slices = min(params.num_slices, total_slices)

    if num_slices == 0:
        return []

    if params.slice_strategy == SliceStrategy.CENTRAL:
        # Central N% of volume (existing medgemma behavior)
        margin = (100 - params.central_percentage) / 200
        start_idx = int(total_slices * margin)
        end_idx = int(total_slices * (1 - margin))

        # Ensure we have enough range
        if end_idx - start_idx < num_slices:
            start_idx = 0
            end_idx = total_slices

        if num_slices == 1:
            indices = [(start_idx + end_idx) // 2]
        else:
            step = (end_idx - start_idx - 1) / (num_slices - 1)
            indices = [int(start_idx + i * step) for i in range(num_slices)]

    elif params.slice_strategy == SliceStrategy.UNIFORM:
        # Evenly spaced across entire volume
        if num_slices == 1:
            indices = [total_slices // 2]
        else:
            step = (total_slices - 1) / (num_slices - 1)
            indices = [int(i * step) for i in range(num_slices)]

    elif params.slice_strategy == SliceStrategy.FIRST_N:
        indices = list(range(num_slices))

    elif params.slice_strategy == SliceStrategy.LAST_N:
        indices = list(range(total_slices - num_slices, total_slices))

    else:
        # Fallback to central
        mid = total_slices // 2
        half = num_slices // 2
        indices = list(range(mid - half, mid - half + num_slices))

    logger.info(f"Extracting slices at indices: {indices} (strategy={params.slice_strategy.value})")

    return [volume_array[i, :, :] for i in indices]


def crop_to_roi(slice_array: np.ndarray, roi: RegionOfInterest) -> np.ndarray:
    """
    Crop one slice to a fractional region of interest.

    The crop happens BEFORE normalization, so the region is windowed against its
    own contents rather than against the whole slice. That is the point of asking
    about a region: a small bright lesion in a mostly dark breast is invisible at
    the slice's own window.

    The result is always at least one pixel in each direction -- a rectangle
    smaller than a pixel is rounded up rather than yielding an empty array that
    would fail deep inside PIL with an unrecognisable error.

    Args:
        slice_array: 2D array shaped (rows, cols)
        roi: Fractional rectangle from the client

    Returns:
        The cropped 2D array
    """
    rows, cols = slice_array.shape[:2]

    left = min(int(round(roi.x * cols)), max(cols - 1, 0))
    top = min(int(round(roi.y * rows)), max(rows - 1, 0))
    right = min(max(int(round((roi.x + roi.width) * cols)), left + 1), cols)
    bottom = min(max(int(round((roi.y + roi.height) * rows)), top + 1), rows)

    logger.info(f"Cropping slice to rows {top}:{bottom}, cols {left}:{right} of {rows}x{cols}")
    return slice_array[top:bottom, left:right]


def slices_to_base64(slice_arrays: list[np.ndarray]) -> list[str]:
    """
    Convert slice arrays to base64-encoded PNG images.

    Args:
        slice_arrays: List of 2D numpy arrays

    Returns:
        List of base64-encoded PNG strings
    """
    base64_images = []

    for slice_arr in slice_arrays:
        # Normalize to 0-255
        normalized = normalize_slice(slice_arr)

        # Convert grayscale to RGB (model expects RGB)
        rgb_array = np.stack([normalized, normalized, normalized], axis=-1)

        # Create PIL Image and encode to base64
        pil_image = Image.fromarray(rgb_array, mode="RGB")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        base64_str = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
        base64_images.append(base64_str)

    return base64_images


async def preprocess_series(
    series_uid: str,
    study_uid: str,
    params: PreprocessingParams,
    wado_base_url: str,
    image_folder: Path,
    selection: SliceSelection | None = None,
) -> list[str]:
    """
    Retrieve and preprocess a DICOM series, returning base64-encoded images.

    Steps:
    1. Retrieve via WADO-RS
    2. Save to temp folder
    3. Read as volume
    4. Select slices -- the instances the message named, else the configured strategy
    5. Normalize and convert to base64 PNG
    6. Cleanup temp files

    Args:
        series_uid: DICOM SeriesInstanceUID
        study_uid: DICOM StudyInstanceUID
        params: Preprocessing parameters (slice count, strategy, etc.)
        wado_base_url: Base URL for WADO-RS retrieval
        image_folder: Base folder for temporary storage
        selection: Per-message slice selection. When it names instances they are
            used verbatim and `params` is ignored; otherwise `params` applies.

    Returns:
        List of base64-encoded PNG image strings

    Raises:
        SliceSelectionError: if the message named slices this series does not hold
    """
    logger.info(f"Preprocessing series {series_uid} from study {study_uid}")

    # Prepare WADO-RS retrieval info
    wado_info = [
        {
            "retrieval_url": f"{wado_base_url}/studies/{study_uid}/series/{series_uid}",
            "study_uid": study_uid,
            "series_uid": series_uid,
        }
    ]

    # Create storage config for temp folder
    storage_config = StorageConfig(image_folder=image_folder, cleanup_on_start=False)

    dicom_folder = None
    try:
        # Retrieve DICOM instances
        logger.info("Retrieving series via WADO-RS...")
        datasets = retrieve_via_wado_rs(wado_info)

        if not datasets:
            raise ValueError(f"No DICOM instances retrieved for series {series_uid}")

        logger.info(f"Retrieved {len(datasets)} DICOM instances")

        # Save to disk
        dicom_folder = save_datasets_to_folder(datasets, series_uid, storage_config)

        # Read as volume, with per-slice identity where the series allows it
        volume, slice_uids = read_dicom_volume_with_instances(dicom_folder)
        volume_array = sitk.GetArrayFromImage(volume)  # Shape: (Z, Y, X)

        logger.info(f"Volume shape: {volume_array.shape} (Z, Y, X)")

        # Select slices: the instances the message named, else the configured strategy
        if selection is not None and selection.sop_instance_uids:
            if not slice_uids:
                raise SliceSelectionError(
                    "This series cannot be addressed slice by slice, so a slice range "
                    "cannot be applied to it. Clear the range to send the configured "
                    "slices instead."
                )
            indices = resolve_selected_indices(slice_uids, selection.sop_instance_uids)
            logger.info(
                f"Sending {len(indices)} selected slices at volume indices {indices} "
                f"(viewer range {selection.range_start}-{selection.range_end} "
                f"of {selection.total_slices})"
            )
            slice_arrays = [volume_array[i, :, :] for i in indices]
        else:
            # The message's own recipe wins over the service's global config.
            slice_arrays = extract_slices(volume_array, effective_params(params, selection))

        # Crop before encoding, and before normalization, so the region is
        # windowed on its own contents.
        if selection is not None and selection.roi is not None:
            slice_arrays = [crop_to_roi(s, selection.roi) for s in slice_arrays]

        # Convert to base64
        base64_images = slices_to_base64(slice_arrays)

        logger.info(f"Preprocessed {len(base64_images)} slices for series {series_uid}")

        return base64_images

    finally:
        # Cleanup temp files
        if dicom_folder and dicom_folder.exists():
            try:
                shutil.rmtree(dicom_folder)
                logger.debug(f"Cleaned up temp folder: {dicom_folder}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp folder {dicom_folder}: {e}")
