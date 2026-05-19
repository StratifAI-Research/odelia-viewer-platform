import os
from shutil import copyfile
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torchio as tio
import torch


def maybe_convert(x):
    if isinstance(x, (pydicom.sequence.Sequence, pydicom.dataset.Dataset)):
        return None  # Don't store complex nested data
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    elif isinstance(x, pydicom.valuerep.DSfloat):
        return float(x)
    elif isinstance(x, pydicom.valuerep.IS):
        return int(x)
    return x


def get(ds, key):
    keyword = ds[key].keyword
    if keyword == "":
        return ds[key].name
    return keyword


def dataset2dict(ds, exclude=['PixelData', 'Overlay Data']):
    return {get(ds, key): maybe_convert(ds[key].value)
            for key in ds.keys()
            if get(ds, key) not in exclude}


def read_metadata(args):
    path_dcm, path_root_data = args
    try:
        # Try to read the DICOM file
        ds = pydicom.dcmread(path_dcm, stop_before_pixels=True)

        # Extract metadata
        meta_dict = dataset2dict(ds)
        meta_dict['_Path'] = str(path_dcm.relative_to(path_root_data))
        return meta_dict

    except Exception as e:
        return None


def sort_dyn(df_dyn):
    # Get n unique Trigger Time and assign index 0 to the smallest and n-1 to the largest trigger time
    df_dyn['TriggerIndex'] = df_dyn['TriggerTime'].rank(method='dense').dropna().astype(int) - 1

    # Verify equal number of slices per dynamic:
    # assert np.unique(df_dyn['TriggerIndex'].value_counts().values).size == 1, "Unequal number of slices per dynamic"
    if np.unique(df_dyn['TriggerIndex'].value_counts().values).size != 1:
        print(f"Excluding {df_dyn.name}, unequal number of slices")
        print(df_dyn['TriggerIndex'].value_counts())
        return None

        # Define the sequence name mapping based on the trigger index
    name_mapping = {0: 'Pre'}
    name_mapping.update({i: f'Post_{i}' for i in range(1, int(df_dyn['TriggerIndex'].max()) + 1)})

    # Assign names based on Trigger Index: {0:'Pre', 1:'Post_1, 2:'Post_2', ...}
    df_dyn['_SequenceName'] = df_dyn['TriggerIndex'].map(name_mapping)

    # Add the total number of dynamic sequences to the DataFrame
    df_dyn['_NumberOfSequences'] = df_dyn['TriggerIndex'].max() + 1

    # Drop the TriggerIndex column if you don't want to keep it
    df_dyn = df_dyn.drop(columns=['TriggerIndex'])

    return df_dyn


def dicom2nii(item, path_data_dicom):
    series_instance_uid, paths_dicoms = item

    # Create temporary folder
    with tempfile.TemporaryDirectory() as temp_dir:
        path_temp_folder = Path(temp_dir)  # Convert to Path object for easy file manipulation

        # Copy files to folder
        for path in paths_dicoms:
            copyfile(path_data_dicom / path, path_temp_folder / Path(path).name)

        # Read DICOM files (assuming the paths are for DICOM files)
        img = tio.ScalarImage(path_temp_folder)  # torchio.Image or ScalarImage for medical imaging
        img.load()  # Load into memory - files can be deleted

    # Create output folder
    study_uid, series_name = series_instance_uid.split('_', 1)  # WARNING: Assumes no "_" in study_uid

    return series_name, img


def dicom_to_unilateral_nifti(dicom_folder: Path, nifti_output_folder=None):
    """
    Receives a dicom folder in which all dicom files are used to create the according nifti file in a unilateral version.
    If parameter nifti_output_folder is not None, the generated nifti file is saved under the provided path.
    nifti_output_folder will be created in case it does not already exist.
    """
    if nifti_output_folder:
        os.makedirs(nifti_output_folder, exist_ok=True)

    # Read all Dicoms
    metadata_list = []
    for path_dcm in dicom_folder.rglob('*.dcm'):
        metadata = read_metadata((path_dcm, dicom_folder))
        metadata_list.append(metadata)

    # Create DataFrame
    metadata_list = [m for m in metadata_list if m is not None]
    df = pd.DataFrame(metadata_list)

    # For T1: seperate dynamic
    df = sort_dyn(df)  # Will add column '_SequenceName' and '_NumberOfSequences'
    df['_SeriesInstanceUID'] = df['SeriesInstanceUID'] + '_' + df['_SequenceName']

    # For T2:
    # df['_SeriesInstanceUID'] = df['SeriesInstanceUID']+'_'+"T2"

    target_shape = (512, 512, 32)
    left_right_split = {
        'right': tio.Crop((256, 0, 0, 0, 0, 0)),
        'left': tio.Crop((0, 256, 0, 0, 0, 0)),
    }

    # DIOCM to TorchIO
    nifties = {}
    series_paths = df.groupby('_SeriesInstanceUID')['_Path'].apply(lambda x: x.to_list())
    for series_path in series_paths.items():
        series_name, img = dicom2nii(series_path, dicom_folder)
        # img.save(f'{series_name}.nii.gz') # Optional save

        # Split
        padding_value = img.data.min().item()  # padding_mode='minimum' calcs minimum per axis, but we want it globally
        crop_or_pad = tio.CropOrPad(target_shape, padding_mode=padding_value)
        cropped_img = crop_or_pad(img)

        # Crop from top and bottom
        thresh = int(np.quantile(cropped_img.data.float(), 0.9))
        foreground_rows = (cropped_img.data > thresh)[0].sum(axis=(0, 2))
        upper_bound = min(max(512 - int(torch.argwhere(foreground_rows).max()) - 10, 0), 256)
        lower_bound = 256 - upper_bound
        height_crop = tio.Crop((0, 0, lower_bound, upper_bound, 0, 0))

        cropped_img = height_crop(cropped_img)

        # seperate left and right
        for side, side_crop in left_right_split.items():
            image_side = side_crop(cropped_img)
            # from os.walk(): root should be patient folder, dirs is empty and files are all dicom files
            if nifti_output_folder:
                image_side.save(f"{nifti_output_folder}/{series_name}_{side}.nii.gz")
            nifties[f"{series_name}_{side}"] = image_side

    return nifties
