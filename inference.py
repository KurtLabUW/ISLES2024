"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-preliminary-docker-evaluation | gzip -c > example-algorithm-preliminary-docker-evaluation.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from glob import glob
from pathlib import Path
import SimpleITK
import json
import subprocess
import os
import sys
from os.path import join
import shutil
import SimpleITK as sitk
import preprocessing

def run():
    INPUT_PATH = Path("/input")
    OUTPUT_PATH = Path("/output")
    RESOURCE_PATH = Path("resources")

    # Read input data.
    ''' TODO- uncomment the image modalities you use in your algorithm.
        In this example, we only use preprocessed_tmax.'''

    # 1) Reading 'raw_data' inputs.
    # 1.1) CT images.

    # non_contrast_ct = load_image_file_as_array(
    #     location=INPUT_PATH / "images/non-contrast-ct",
    # )
    # ct_angiography = load_image_file_as_array(
    #     location=INPUT_PATH / "images/ct-angiography",
    # )
    # perfusion_ct = load_image_file_as_array(
    #     location=INPUT_PATH / "images/perfusion-ct",
    # )
    #
    # # 1.2) Perfusion maps.
    # tmax_parameter_map = load_image_file_as_array(
    #     location=INPUT_PATH / "images/tmax-parameter-map",
    # )
    # cbf_parameter_map = load_image_file_as_array(
    #     location=INPUT_PATH / "images/cbf-parameter-map",
    # )
    # cbv_parameter_map = load_image_file_as_array(
    #     location=INPUT_PATH / "images/cbv-parameter-map",
    # )
    # mtt_parameter_map = load_image_file_as_array(
    #     location=INPUT_PATH / "images/mtt-parameter-map",
    # )


    # # 2) Reading 'derivatives' inputs.
    # # 2.1) CT images.
    #
    # preprocessed_ct_angiography = load_image_file_as_array(
    #     location=INPUT_PATH / "images/preprocessed-CT-angiography",
    # )
    # preprocessed_perfusion_ct = load_image_file_as_array(
    #     location=INPUT_PATH / "images/preprocessed-perfusion-ct",
    # )
    #
    # # 2.2) Perfusion maps.
    # preprocessed_tmax_map = load_image_file_as_array(
    #     location=INPUT_PATH / "images/preprocessed-tmax-map",
    # )
    # preprocessed_cbf_map = load_image_file_as_array(
    #     location=INPUT_PATH / "images/preprocessed-cbf-map",
    # )
    # preprocessed_cbv_map = load_image_file_as_array(
    #     location=INPUT_PATH / "images/preprocessed-cbv-map",
    # )
    
    # preprocessed_mtt_map = load_image_file_as_array(
    #     location=INPUT_PATH / "images/preprocessed-mtt-map",
    # )

    # 3) Reading 'phenotype' (clinical 'baseline' tabular data)
    #acute_stroke_clinical_information = json.load(open(INPUT_PATH / "acute-stroke-clinical-information.json"))


    # using resources.
    # with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
    #     print(f.read())
    # Prediction scripts come below.
    ################################################################################################################
    #################################### here comes your predictions algorithm  ####################################
    #_show_torch_cuda_info() # comment out to test pytorch/cuda
    predict_infarct() # todo -function to be updated by you!
    ################################################################################################################

    # Save your output
    # write_array_as_image_file(
    #     location=OUTPUT_PATH / "images/stroke-lesion-segmentation",
    #     array=stroke_lesion_segmentation,
    # )

    return 0

def nnunet_dataset_conversion(data_path):
    out_base = "/tmp/raw"
    if not os.path.exists(out_base):
        os.makedirs(out_base)
    move_file(glob(str(join(data_path, "preprocessed-cbf-map", "*.mha")))[0], "/tmp/raw/isles0000_0000.nii.gz")
    move_file(glob(str(join(data_path, "preprocessed-cbv-map", "*.mha")))[0], "/tmp/raw/isles0000_0001.nii.gz")
    move_file(glob(str(join(data_path, "preprocessed-mtt-map", "*.mha")))[0], "/tmp/raw/isles0000_0002.nii.gz")
    move_file(glob(str(join(data_path, "preprocessed-tmax-map", "*.mha")))[0], "/tmp/raw/isles0000_0003.nii.gz")
    move_file(glob(str(join(data_path, "preprocessed-CT-angiography", "*.mha")))[0], "/tmp/raw/isles0000_0004.nii.gz")
    
    return "/tmp/raw"

def move_file(source, dest):
    img = sitk.ReadImage(source)
    sitk.WriteImage(img, dest)

def nnunet(convertedInput, output_path):
    output_path = output_path.rstrip("/")
    
    nnUNetRun = "nnUNet.nnunetv2.inference.predict_from_raw_data"

    #subprocess.run([sys.executable, "-m", nnUNetRun, '-d', '150', '-i', convertedInput, '-o', f"{output_path}", '-f', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-tr', 'nnUNetTrainer', '-c', '3d_fullres', '-p', 'nnUNetResEncUNetLPlans', "--save_probabilities", "-device", "cpu"], check=True)
    subprocess.run([sys.executable, "-m", nnUNetRun, '-d', '150', '-i', convertedInput, '-o', f"{output_path}", '-f', '0', '-tr', 'nnUNetTrainer', '-c', '3d_fullres', '-p', 'nnUNetResEncUNetLPlans', "--save_probabilities", "-device", "cpu"], check=True)


def predict_infarct():
    data = nnunet_dataset_conversion(join(Path("/input"), "images"))
    preprocessed = preprocessing.run_preprocessing(data, "/tmp/preprocessed")
    nnunet(preprocessed, "/tmp/inference")
    img = sitk.ReadImage("/tmp/inference/isles0000.nii.gz")
    sitk.WriteImage(img, join(Path("/output"), "images/stroke-lesion-segmentation/output.mha"), useCompression=True)

def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

if __name__ == "__main__":
    raise SystemExit(run())
