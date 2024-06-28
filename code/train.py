import yaml
import torch
from utils.helper import set_seed
import os
import re
from dataloader_transforms import XrayDataset
from architecture_selector import model_selector
from optimizers import optimizer_selector
import copy
from datetime import datetime
import json
import shutil
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument('--checkpoint', type=str, default="", action = "store")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML file
        config = yaml.safe_load(file)
    return config

# Replace 'config.yaml' with the path to your YAML file
config_file = 'model_params.yaml'
if args.checkpoint:
    config_file = os.path.join(args.checkpoint, config_file)
config = read_yaml_config(config_file)

print(config)

### seeding

set_seed(config["seed"])
ROOT_FOLDER = config["data_folder"]

### prepare data

ts = []
for file in os.listdir(ROOT_FOLDER):
    if file.endswith(".pt") and re.match(r"^[Xy]_\d+_\d+\.pt$", file):
        ts.append(file)
    if file.endswith(".pt") and re.match(r"^[Xy]_\d+\.pt$", file):
        ts.append(file)
    
ts.sort()
# simple check all x, y are paired
assert len(ts)%2 == 0

ts_x = ts[:len(ts)//2]
ts_y = ts[len(ts)//2:]

X = torch.cat([torch.load(os.path.join(ROOT_FOLDER, t)) for t in ts_x])
y = torch.cat([torch.load(os.path.join(ROOT_FOLDER, t)) for t in ts_y])


# mask out y's that have missing values

mask = torch.isinf(y).any(dim=1)

if mask.sum()>0:
    print("Missing data detected. Images that are not properly annotated are dropped")

X = X[~mask]
y = y[~mask]


# train val split
test_ratio = config["training"]["test_ratio"]

num_samples = X.size(0)
train_size = int( (1-test_ratio) * num_samples)
test_size = num_samples - train_size

shuffled_indices = torch.randperm(num_samples)

train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

# Use indices to create training and test sets
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]



train_set, val_set = XrayDataset(X_train, y_train), XrayDataset(X_test,y_test,augmentation=None)

model = model_selector(model_name=config["model"]["type"], source_domain=config["model"]["source_domain"]).to(DEVICE)

bs = config["training"]["batch_size"]

train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False)  # drop_last?

criterion = torch.nn.MSELoss() # can be changed


optimizer = optimizer_selector(optim_name=config["optimizer"]["type"], lr = config["optimizer"]["learning_rate"], model=model)

if args.checkpoint:
    checkpoint = torch.load(os.path.join(args.checkpoint, "model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

### logging
training_loss = []
validation_loss =[]
best_val = float("inf")
best_state = None
best_metrics = None
best_epoch = None
best_state = None
best_opt_state = None

### training loop 

###### a hyperparameter search space can be setup here

num_epochs = config["training"]["epochs"]

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    iterations = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        if len(inputs) == bs:
            running_loss += loss.item()
        # print(outputs[0], running_loss)
        else:
            running_loss = running_loss + loss.item()*len(labels)/bs 


    # Print average training loss for this epoch
    training_loss.append(running_loss / len(train_loader))


    # Validation phase
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if len(inputs) == bs:
                total_loss += loss.item()
            else: 
                total_loss = total_loss + loss.item()*len(targets)/bs 
            
    if total_loss / len(val_loader) < best_val:
        best_val = total_loss / len(val_loader) 
        best_metrics  = (training_loss[-1], best_val)
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        best_opt_state = copy.deepcopy(optimizer.state_dict())
        print("***************")


    # Print average validation loss for this epoch
    validation_loss.append(total_loss / len(val_loader))
    print(f"Epoch {epoch+1}, Train Loss: {training_loss[-1]}, Val Loss: {validation_loss[-1]}")

print("Training is complete")

print(best_metrics, best_epoch)

### can add additional info to the string
current_datetime = datetime.now().strftime('%m%d%H')


#torch.save(best_state, f"{current_datetime}-{config['model']['type']}-{num_epochs}ep.pth")

model_folder = f"{current_datetime}-{config['model']['type']}-{num_epochs}ep"
model_file = "model.pth"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)
else: 
    print("the model folder already exists")

# Construct the new path for the file
new_file_path = os.path.join(model_folder, model_file)

# Copy the file

torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_state,
            'optimizer_state_dict': best_opt_state,
            }, new_file_path)


### logging additional info
logs = {'best_metrics': best_metrics, "best_epoch": best_epoch, 'training_loss': training_loss, 'val_loss': validation_loss}

# Specify the filename
filename = os.path.join(model_folder, "log.json")

# Write the dictionary to a file
with open(filename, 'w') as f:
    json.dump(logs, f)

print(f"Dictionary saved to {filename}")

new_file_path = os.path.join(model_folder, config_file)

# Copy the file
shutil.copy(config_file, new_file_path)

print(f"File {config_file} has been copied to {new_file_path}")