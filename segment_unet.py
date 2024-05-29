import os
import glob
from dataset import get_data_dicts
import matplotlib.pyplot as plt
from monai.utils import first, set_determinism
#from monai.transforms import LoadNiftid
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImage,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Resized,Resize,
    SaveImaged,
    ScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    DivisiblePadd,
    RandAffined,
    RandRotated,
    RandGaussianNoised
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import tempfile
import shutil
from datetime import datetime


# Create directories function
def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Define directories within your home directory
home_dir = "/nobackup/scxcw/dataset_cmr/segmentation_unet/"
model_dir = os.path.join(home_dir, "models")
plot_dir = os.path.join(home_dir, "plot")
plot_pred_dir = os.path.join(home_dir, "plot_pred")

# Create the directories if they don't exist
create_dir_if_not_exists(model_dir)
create_dir_if_not_exists(plot_dir)
create_dir_if_not_exists(plot_pred_dir)




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


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
batch_size = 32

# Define the transformations for training data
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
  #  	Resize(spatial_size=(128, 128)),
     #   Orientationd(keys=["image", "label"], axcodes="RAS"),
        Resized(keys=["image", "label"], spatial_size=(128, 128)),
        ScaleIntensityd(keys=["image", "label"]),
     #   AsDiscreted(keys=["label"], to_onehot=2) 
     #   DivisiblePadd(keys=["image", "label"], k=16)
    ]
)

# Define the transformations for validation data
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Resized(keys=["image", "label"], spatial_size=(128, 128)),
        DivisiblePadd(keys=["image", "label"], k=16)
    ]
)

augm_transforms = Compose(
    [
        RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10), 
        RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
        RandGaussianNoised(keys='image', prob=0.5),
    ]
)

# Images -> transform

# Images -> transform + data augmentation
# augm_ds= CacheDataset(data=train_files, transform=[train_transforms, augm_transforms])

# Batch size -> whole dataset -> (data) + (data augmentation) =  images per epoch
# this means 1 batch = 32 images
# train_ds = ConcatDataset([train_ds, augm_ds])





# Create datasets and data loaders
train_ds = CacheDataset(data=train_files, transform=train_transforms)
val_ds = CacheDataset(data=val_files, transform=train_transforms)
test_ds = CacheDataset(data=test_files, transform=train_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=1)

# implementing the early stopping
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
# Define the UNet model for 2D images
model = UNet(
    spatial_dims=2, # 2D images
    in_channels=1, # input channels
    out_channels=1, # output channels for soft labeling
    channels=(16, 32, 64, 128, 256), # channels in each layer
    strides=(2, 2, 2, 2), # strides for downsampling
    num_res_units=2, # number of residual units
    dropout=0.2, # dropout rate
    norm='batch', # normalization type
).to(device)



loss_function = DiceCELoss(sigmoid = True)
optimizer = torch.optim.AdamW(model.parameters(), 1e-4, amsgrad = True)
early_stopper = EarlyStopper(patience=3, min_delta=0.3)

total_time = 0
max_epochs = 100
val_interval = 1  # not a large dataset, so it is fine
print_interval = 20
epoch_loss_values = []
losses_validation = []

for epoch in range(max_epochs):
    start_time = datetime.now()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()  # tell Dropout and BatchNorm to work because it is training
    epoch_loss = 0
    step = 0
    
    # getting data for each batch
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        
        # normal pipeline
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # accumulate loss
        epoch_loss += loss.item()
        if step % print_interval == 0:
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
    
    # measuring time        
    actual_time = datetime.now() - start_time
    print(f"time to train this epoch: {actual_time}")
    total_time += actual_time.total_seconds()
    # saving the loss for the actual epoch
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    

    if (epoch + 1) % val_interval == 0:
        model.eval()  # tell Dropout and BatchNorm to "turn off" because I am evaluating the model
        with torch.no_grad():
            loss_val = 0
            val_steps = 0
            for val_data in val_loader:
                val_steps += 1
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                # get the loss to the validation set
                outputs = model(val_inputs)
                loss_val += loss_function(outputs, val_labels).item()
            
            loss_val_avg = loss_val / val_steps
            losses_validation.append(loss_val_avg)
            print(f"validation average loss: {loss_val_avg:.4f}")
            if early_stopper.early_stop(loss_val_avg):
                print("early stopped!")
                break

# Save the model
# Define the directory path


# Check if the directory exists, if not, create it


torch.save(model.state_dict(), os.path.join(model_dir, "unet2dAdamwDicece.pth"))


# After the training and validation loop
# Save the training and validation loss graphs

plt.figure("train", (12, 6))

# Plot Epoch Average Loss
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.title("Validation Loss")
x = [val_interval * (i + 1) for i in range(len(losses_validation))]
y = losses_validation
plt.xlabel("epoch")
plt.plot(x, y)

# Save the figure
plt.savefig(os.path.join(plot_dir, "loss_graph.png"))
plt.close()


# Containers for images, model outputs, and labels
images = []
outputs = []
labels = []

# Process the data in the test_loader
for test_data in test_loader:
    test_inputs, test_labels = (
        test_data["image"].to(device),
        test_data["label"].to(device),
    )
    images.append(test_inputs)
    with torch.no_grad():
        outputs.append(model(test_inputs))
    labels.append(test_labels)

# Number of images per figure
images_per_figure = 10

# Loop over all images and save them in separate figures
for start_idx in range(0, len(images), images_per_figure):
    end_idx = min(start_idx + images_per_figure, len(images))
    num_images = end_idx - start_idx
    
    plt.figure(figsize=(15, 5 * num_images))

    for i in range(num_images):
        idx = start_idx + i
        output = outputs[idx]
        label = labels[idx][0]
        image = images[idx][0]

        # Plot the Model Output
        plt.subplot(num_images, 3, 3 * i + 1)
        plt.imshow(torch.argmax(output.cpu(), dim=1)[0, :, :], cmap="gray")
        plt.title(f'Model Output - Image {idx + 1}')

        # Plot the Label
        plt.subplot(num_images, 3, 3 * i + 2)
        plt.imshow(label.cpu()[0, :, :], cmap="gray")
        plt.title(f'Label - Image {idx + 1}')

        # Plot the Input Image
        plt.subplot(num_images, 3, 3 * i + 3)
        plt.imshow(image.cpu()[0, :, :], cmap="gray")
        plt.title(f'Input Image - Image {idx + 1}')

    # Adjust layout for better visualization
    plt.tight_layout()

    # Save the figure instead of displaying it
plt.savefig(os.path.join(plot_pred_dir, f'single_image_prediction_{start_idx // images_per_figure + 1}.png'))
plt.close()
