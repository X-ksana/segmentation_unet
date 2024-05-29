import os
import glob

def get_data_dicts(data_dir):
    # Get all image and label files
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_mid_slice_image.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(data_dir, "*_mid_slice_label.nii.gz")))

    # Create a dictionary matching images with labels based on common prefix
    data_dicts = []
    for image_file in image_files:
        # Extract the common prefix
        prefix = os.path.basename(image_file).replace("_mid_slice_image.nii.gz", "")
        # Find the corresponding label file
        label_file = os.path.join(data_dir, prefix + "_mid_slice_label.nii.gz")
        if label_file in label_files:
            data_dicts.append({"image": image_file, "label": label_file})

    return data_dicts

"""
data_dir = "/nobackup/scxcw/dataset_cmr/"
train_dir = os.path.join(data_dir, "MCMV_Train")
val_dir = os.path.join(data_dir, "MCMV_Valid")
test_dir = os.path.join(data_dir, "MCMV_Test")

# Get data dictionaries for each dataset
train_files = get_data_dicts(train_dir)
val_files = get_data_dicts(val_dir)
test_files = get_data_dicts(test_dir)

print("Training files:", train_files)
print("Validation files:", val_files)
print("Testing files:", test_files)



"""
