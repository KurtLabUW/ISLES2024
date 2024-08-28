import os
import glob
import shutil
import nibabel as nib
import numpy as np
from skimage import exposure
from concurrent.futures import ProcessPoolExecutor

def apply_histogram_equalization_custom_range(input_file, output_file, min_intensity, max_intensity):
    img = nib.load(input_file)
    data = img.get_fdata()

    # Clip the data to the custom intensity range
    data_clipped = np.clip(data, min_intensity, max_intensity)
    data_clipped[data == 0] = 0
    # Normalize the data to [0, 1] range
    data_clipped[data_clipped > 0.0001] -= min_intensity
    data_normalized = data_clipped / (max_intensity - min_intensity)

    # Apply 3D histogram equalization
    equalized_data = exposure.equalize_hist(data_normalized, mask=(data_normalized > 0.0001))
    
    equalized_data[data_normalized < 0.0001] = 0
     
    # Save the result as a new NIfTI file
    equalized_img = nib.Nifti1Image(equalized_data, img.affine, img.header)
    nib.save(equalized_img, output_file)
    print(f"Saved equalized image to {output_file}")

def process_training_case(case_identifier, input_dir, output_dir, intensity_ranges):
    case_files = sorted(glob.glob(os.path.join(input_dir, f"{case_identifier}_*.nii.gz")))

    # Process each channel based on its 4-digit identifier
    for input_file in case_files:
        channel_id = input_file.split('_')[-1].split('.')[0]
        if channel_id in intensity_ranges:
            min_intensity, max_intensity = intensity_ranges[channel_id]
            output_file = os.path.join(output_dir, f"{case_identifier}_{channel_id}.nii.gz")
            apply_histogram_equalization_custom_range(input_file, output_file, min_intensity, max_intensity)
            
        else:
            # If the channel is not in intensity_ranges, simply copy it
            output_file = os.path.join(output_dir, f"{case_identifier}_{channel_id}.nii.gz")
            shutil.copy(input_file, output_file)
            print(f"Copied {input_file} to {output_file}")

def process_all_cases(input_dir, output_dir, intensity_ranges):
    case_identifiers = {os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(input_dir, "*_*.nii.gz"))}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_training_case, case_identifier, input_dir, output_dir, intensity_ranges)
                   for case_identifier in case_identifiers]
        for future in futures:
            future.result()  # This will raise any exceptions encountered during processing

def run_preprocessing(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Define custom intensity ranges for each modality/channel
    intensity_ranges = {
        '0000': (0, 35),   # Example range for channel 0000 (e.g., T1)
        '0001': (0, 10), # Example range for channel 0001 (e.g., T2)
        '0002': (0, 20),  # Example range for channel 0002
        '0003': (0, 7),
        '0004': (0, 90),
    }

    process_all_cases(input_dir, output_dir, intensity_ranges)
    print("Processing complete.")
    return output_dir
    
if __name__ == "__main__":
    print("Processing complete.")
