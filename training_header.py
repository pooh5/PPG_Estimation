import os
import pickle
import h5py
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
from loguru import logger

import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR

from collections import defaultdict

PROJECT_FOLDER = "/home/pooja/tpp_ppg_training"
PICKLES_DIR = "/home/pooja/tpp_ppg_training/pickles"
TPP_PPG_FNAME = "/media/local/ps_datasets/ppg_abp_tpp_v2.hd5"
_DEVICE = "cuda:1"
DEVICE = torch.device(_DEVICE)

# Create directory if it doesn't exist
if not os.path.exists(PICKLES_DIR):
    os.makedirs(PICKLES_DIR)

logger.remove()
logger.add(os.path.join(PROJECT_FOLDER, "logs", "training.log"), rotation="10 MB")

def reset_all_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

# ==================================================================================================== #
#                  FUNCTIONS FOR DEFINING TRAIN/TUNE/TEST SPLITS AND MRN/CSN MAPPINGS                  #
# ==================================================================================================== #
def _get_mrn_csn_map():
    pickle_path = os.path.join(PICKLES_DIR, "mrn_to_csns.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            mrn_to_csns = pickle.load(f)
    else:
        with h5py.File(TPP_PPG_FNAME, 'r') as f:
            mrn_to_csns = {mrn: list(f[mrn].keys()) for mrn in f.keys()}
        with open(pickle_path, 'wb') as f:
            pickle.dump(mrn_to_csns, f)

    return mrn_to_csns

def _get_train_tune_test_mrns():
    pickle_path = os.path.join(PICKLES_DIR, "train_tune_test_mrns.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            train_mrns, tune_mrns, test_mrns = pickle.load(f)
    else:
        mrn_to_csns = _get_mrn_csn_map()
        all_mrns = list(mrn_to_csns.keys())
        np.random.shuffle(all_mrns)
        # 80% train, 10% tune, 10% test
        train_mrns = all_mrns[:int(0.8*len(all_mrns))]
        tune_mrns = all_mrns[int(0.8*len(all_mrns)):int(0.9*len(all_mrns))]
        test_mrns = all_mrns[int(0.9*len(all_mrns)):]

        with open(pickle_path, 'wb') as f:
            pickle.dump((train_mrns, tune_mrns, test_mrns), f)

    return train_mrns, tune_mrns, test_mrns

def get_train_mrn_csn_map():
    mrn_to_csns = _get_mrn_csn_map()
    train_mrns, _, _ = _get_train_tune_test_mrns()
    return {mrn: mrn_to_csns[mrn] for mrn in train_mrns}

def get_tune_mrn_csn_map():
    mrn_to_csns = _get_mrn_csn_map()
    _, tune_mrns, _ = _get_train_tune_test_mrns()
    return {mrn: mrn_to_csns[mrn] for mrn in tune_mrns}

def get_test_mrn_csn_map():
    mrn_to_csns = _get_mrn_csn_map()
    _, _, test_mrns = _get_train_tune_test_mrns()
    return {mrn: mrn_to_csns[mrn] for mrn in test_mrns}


def count_hours(mrn, csn):
    with h5py.File(TPP_PPG_FNAME, 'r') as f:
        return f[mrn][csn]['abp'].shape[0]/60


# ==================================================================================================== #
#                       FUNCTIONS FOR GETTING A DATASET INDEX [(MRN, CSN, IND)]                        #
# ==================================================================================================== #
def create_dataset_index(mrn_to_csns, condition, savename, force_regen=False):
    if os.path.exists(os.path.join(PICKLES_DIR, savename)):
        if not force_regen:
            return
        else:
            os.remove(os.path.join(PICKLES_DIR, savename))

    dataset_index = []
    with h5py.File(TPP_PPG_FNAME, 'r') as f:
        for mrn, csn_list in tqdm(mrn_to_csns.items(), total=len(mrn_to_csns), desc="Creating dataset index"):
            patient_index = []
            for csn in csn_list:
                mask = condition(f[mrn][csn])
                for ind in np.argwhere(mask).flatten():
                    patient_index.append((mrn, csn, ind))

            if len(patient_index) > 0:
                dataset_index.append(patient_index)


    with open(os.path.join(PICKLES_DIR, savename), 'wb') as f:
        pickle.dump(dataset_index, f)

def load_dataset_index(savename):
    if not os.path.exists(os.path.join(PICKLES_DIR, savename)):
        raise FileNotFoundError(f"Could not find {savename} in {PICKLES_DIR}")

    with open(os.path.join(PICKLES_DIR, savename), 'rb') as f:
        return pickle.load(f)


# ==================================================================================================== #
#                                          EXAMPLE CONDITIONS                                          #
# ==================================================================================================== #

def rmse_lt_3_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10(hdf_group):
    if 'fit_rmse' not in hdf_group or 'p_crit' not in hdf_group or 'map' not in hdf_group or 'pp' not in hdf_group:
        return False

    num_minutes = hdf_group['ppg'].shape[0]
    if num_minutes < 100:
        return False

    pcrit = hdf_group['p_crit'][:]
    fit_rmse = hdf_group['fit_rmse'][:]
    maps = hdf_group['map'][:]
    pp = hdf_group['pp'][:]
    return (fit_rmse < 3) & (pcrit > 0) & (pcrit < maps) & (pp > 10)

def rmse_lt_2_and_pcrit_gt_0_pcrit_lt_100_pp_gt_10(hdf_group):
    if 'fit_rmse' not in hdf_group or 'p_crit' not in hdf_group or 'map' not in hdf_group or 'pp' not in hdf_group:
        return False

    num_minutes = hdf_group['ppg'].shape[0]
    if num_minutes < 100:
        return False

    pcrit = hdf_group['p_crit'][:]
    fit_rmse = hdf_group['fit_rmse'][:]
    maps = hdf_group['map'][:]
    pp = hdf_group['pp'][:]
    return (fit_rmse < 2) & (pcrit > 0) & (pcrit < 100) & (pp > 10) & (pcrit < maps)


def rmse_lt_1_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10(hdf_group):
    if 'fit_rmse' not in hdf_group or 'p_crit' not in hdf_group or 'map' not in hdf_group or 'pp' not in hdf_group:
        return False

    num_minutes = hdf_group['ppg'].shape[0]
    if num_minutes < 100:
        return False

    pcrit = hdf_group['p_crit'][:]
    fit_rmse = hdf_group['fit_rmse'][:]
    maps = hdf_group['map'][:]
    pp = hdf_group['pp'][:]
    return (fit_rmse < 1) & (pcrit > 0) & (pcrit < maps) & (pp > 10)

def r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10(hdf_group):
    if 'r2' not in hdf_group or 'p_crit' not in hdf_group or 'map' not in hdf_group or 'pp' not in hdf_group:
        return False

    num_minutes = hdf_group['ppg'].shape[0]
    if num_minutes < 100:
        return False

    pcrit = hdf_group['p_crit'][:]
    r2 = hdf_group['r2'][:]
    maps = hdf_group['map'][:]
    pp = hdf_group['pp'][:]
    return (r2 > 0.5) & (pcrit > 0) & (pcrit < maps) & (pp > 10)

def r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120(hdf_group):
    if 'r2' not in hdf_group or 'p_crit' not in hdf_group or 'map' not in hdf_group or 'pp' not in hdf_group:
        return False

    num_minutes = hdf_group['ppg'].shape[0]
    if num_minutes < 100:
        return False

    pcrit = hdf_group['p_crit'][:]
    r2 = hdf_group['r2'][:]
    maps = hdf_group['map'][:]
    pp = hdf_group['pp'][:]
    return (r2 > 0.5) & (pcrit > 0) & (pcrit < maps) & (pp > 10) & (maps < 120)

def r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120_pcrit_lt_100(hdf_group):
    if 'r2' not in hdf_group or 'p_crit' not in hdf_group or 'map' not in hdf_group or 'pp' not in hdf_group:
        return False

    num_minutes = hdf_group['ppg'].shape[0]
    if num_minutes < 100:
        return False

    pcrit = hdf_group['p_crit'][:]
    r2 = hdf_group['r2'][:]
    maps = hdf_group['map'][:]
    pp = hdf_group['pp'][:]
    return (r2 > 0.5) & (pcrit > 0) & (pcrit < maps) & (pp > 10) & (maps < 120) & (pcrit<100)

# ==================================================================================================== #
#                                            DATASET OBJECT                                            #
# ==================================================================================================== #

@torch.no_grad()
def ppg_norm(ppg_wf):
    mean = ppg_wf.mean()
    #min_group = ppg_means < 33 <-> Mean: 31.1961
    #mid_group = (ppg_means >= 33) & (ppg_means < 38) <-> Mean: 36.4943
    #max_group = ppg_means >= 38 <-> Mean: 39.7689
    #return ppg_wf/35

    if mean < 33:
        return ppg_wf/31.1961 - 1.0
    if mean < 38:
        return ppg_wf/36.4943 - 1.0
    return ppg_wf/39.7689 - 1.0

#DATA_MEANS = {"map": 73.40, "sbp": 126.73, "dbp": 49.01, "pp": 78.23, "hr": 74.27} #, "p_crit": 31.05}
#DATA_STD = {"map": 7.92, "sbp": 20.35, "dbp": 7.37, "pp": 20.05, "hr": 25.97} #, "p_crit": 12.55}

DATA_MEANS = {"map": 76.73, "sbp": 118.09, "dbp": 56.11, "pp": 62.47, "hr": 76.95} #, "pcrit": 40.54}
DATA_STD = {"map": 11.21, "sbp": 19.53, "dbp": 10.23, "pp": 18.46, "hr": 17.31} #, "pcrit": 15.51}

@torch.no_grad()
def wf_vital_norm(val, key):
    if key == "ppg":
        return ppg_norm(val)

    if key not in DATA_MEANS:
        return val

    return (val - DATA_MEANS[key]) / DATA_STD[key]

class TPP_PPG_dataset(Dataset):
    def __init__(self, dataset_index, level, keys_to_load=None, percentage=100, csns_to_skip=None):
        assert level in ['segment', 'patient']
        assert percentage > 0 and percentage <= 100

        if keys_to_load is None:
            keys_to_load = ['ppg', 'p_crit']
        self.keys_to_load = keys_to_load

        self.level = level
        self.csns_to_skip = set(csns_to_skip or [])

        # Flatten or keep hierarchical index
        if self.level == 'segment':
            self.dataset_index = []
            for patient_index in dataset_index:
                self.dataset_index.extend(patient_index)
        else:
            self.dataset_index = dataset_index

        # Filter based on CSN
        if self.csns_to_skip:
            if self.level == 'segment':
                self.dataset_index = [entry for entry in self.dataset_index if entry[1] not in self.csns_to_skip]
            else:
                self.dataset_index = [
                    [entry for entry in patient_entries if entry[1] not in self.csns_to_skip]
                    for patient_entries in self.dataset_index
                ]
                self.dataset_index = [p for p in self.dataset_index if p]  # remove empty ones

        # Shuffle and reduce by percentage
        shuffle_inds = np.random.permutation(len(self.dataset_index))
        self.dataset_index = [self.dataset_index[ind] for ind in shuffle_inds]
        if percentage < 100:
            self.dataset_index = self.dataset_index[:int(percentage / 100 * len(self.dataset_index))]

        self.open_file = None

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, idx):
        if self.open_file is None:
            self.open_file = h5py.File(TPP_PPG_FNAME, 'r', swmr=True)

        if self.level == 'segment':
            mrn, csn, ind = self.dataset_index[idx]
        else:
            i = np.random.choice(len(self.dataset_index[idx]))
            mrn, csn, ind = self.dataset_index[idx][i]

        hdf_group = self.open_file[mrn][csn]
        returns = {key: torch.tensor(hdf_group[key][ind, ...], dtype=torch.float32) for key in self.keys_to_load}

        return tuple(returns.values())



# ==================================================================================================== #
#                                                MODEL                                                 #
# ==================================================================================================== #
class tpp_model(nn.Module):
    def __init__(self, cnn_encoder, mlp):
        super(tpp_model, self).__init__()
        self.cnn_encoder = cnn_encoder
        self.mlp = mlp

    def forward(self, ppg, *discretes):
        ppg = self.cnn_encoder(ppg)
        discretes = [d.unsqueeze(1) if len(d.shape) == 1 else d for d in discretes]
        x = torch.cat([ppg, *discretes], dim=1)
        x = self.mlp(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()

        # Convolutional Layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)

        # Instance norm layer
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)

        # Activation Function (e.g., ReLU)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.activation(x)
        return x

class MiniRes(nn.Module):
    def __init__(self, size):
        super(MiniRes, self).__init__()
        self.instance_norm = nn.InstanceNorm1d(size, affine=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.instance_norm(x)
        x = self.activation(x)
        return x
    

class ResBlockV1(nn.Module):
    def __init__(self, in_channels, out_1, kernel_1, out_2, kernel_2, max_kern, stride=1):
        super(ResBlockV1, self).__init__()

        self.max_pool = nn.MaxPool1d(max_kern, stride=max_kern, padding=max_kern//2)
        self.conv1x1 = nn.Conv1d(in_channels, out_2, 1, stride=stride)

        self.conv1 = ConvBlock(in_channels, out_1, kernel_1, stride=stride)
        self.conv2 = nn.Conv1d(out_1, out_2, kernel_2, padding=kernel_2//2)

    def forward(self, x):
        skip = self.max_pool(x)
        skip = self.conv1x1(skip)

        main = self.conv1(x)
        main = self.conv2(main)

        # fin=main+skip ##

        return main + skip

class ResBlockV2(nn.Module):
    def __init__(self, in_channels, out_1, kernel_1, out_2, kernel_2, max_kern, stride=1):
        super(ResBlockV2, self).__init__()

        self.mini_res = MiniRes(in_channels)

        self.max_pool = nn.MaxPool1d(max_kern, stride=max_kern, padding=max_kern//2)
        self.conv1x1 = nn.Conv1d(in_channels, out_2, 1, stride=stride)

        self.conv1 = ConvBlock(in_channels, out_1, kernel_1, stride=stride)
        self.conv2 = nn.Conv1d(out_1, out_2, kernel_2, padding=kernel_2//2)

    def forward(self, x):
        skip = self.max_pool(x)
        skip = self.conv1x1(skip)

        main = self.mini_res(x)
        main = self.conv1(main)
        main = self.conv2(main)

        return main + skip

class ResBlockV3(nn.Module):
    def __init__(self, in_channels, out_1, kernel_1, out_2, kernel_2, max_kern, stride=1):
        super(ResBlockV3, self).__init__()

        self.mini_res = MiniRes(in_channels)

        self.max_pool = nn.MaxPool1d(max_kern, stride=max_kern, padding=max_kern//2)
        self.conv1x1 = nn.Conv1d(in_channels, out_2, 1, stride=stride)

        self.conv1 = ConvBlock(in_channels, out_1, kernel_1, stride=stride)
        self.conv2 = nn.Conv1d(out_1, out_2, kernel_2, padding=kernel_2//2)

        self.mini_res_end = MiniRes(out_2)

    def forward(self, x):
        skip = self.max_pool(x)
        skip = self.conv1x1(skip)
        main = self.mini_res(x)
        main = self.conv1(main)
        main = self.conv2(main)
        x = self.mini_res_end(main+skip)

        return x

class MyFirstResNet(nn.Module):
    def __init__(self, in_channels=1, num_discrete=0):
        super(MyFirstResNet, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_channels, 64, 15),
            ResBlockV1(64, 64, 15, 64, 15, 1, 4),
            ResBlockV2(64, 128, 15, 128, 15, 1, 4),
            ResBlockV2(128, 128, 15, 128, 15, 1, 4),
            ResBlockV3(128, 256, 15, 256, 15, 1, 4)
        )
        self.gap = nn.AvgPool1d(15)
        self.projection = nn.Linear(256 + num_discrete, 128)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, ppg, *discretes):
        features = self.layers(ppg)
        features = self.gap(features)
        features = features.view(features.size(0), -1)  # Flattening

        discretes = [d.unsqueeze(1) if len(d.shape) == 1 else d for d in discretes]  
        combined_input = torch.cat([features, *discretes], dim=1)

        output = self.projection(combined_input)
        output = self.mlp(output)
        return output
    
class MyFirstResNet_V2(nn.Module):
    def __init__(self, in_channels=1, num_discrete=0):
        super(MyFirstResNet_V2, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_channels, 64, 15),
            ResBlockV1(64, 64, 31, 64, 15, 1, 2),
            ResBlockV2(64, 128, 51, 128, 51, 1, 2),
            ResBlockV2(128, 128, 101, 128, 101, 1, 4),
            ResBlockV3(128, 256, 201, 256, 201, 1, 4),
            ResBlockV3(256, 256, 401, 256, 401, 1, 4)
        )
        self.gap = nn.AvgPool1d(15)
        self.projection = nn.Linear(256 + num_discrete, 128)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, ppg, *discretes):
        features = self.layers(ppg)
        features = self.gap(features)
        features = features.view(features.size(0), -1)  # Flattening

        discretes = [d.unsqueeze(1) if len(d.shape) == 1 else d for d in discretes]  
        combined_input = torch.cat([features, *discretes], dim=1)

        output = self.projection(combined_input)
        output = self.mlp(output)
        return output
    


# ==================================================================================================== #
#                                            TRAINING LOOP                                             #
# ==================================================================================================== #


def training_loop(
        model,
        train_loader, val_loader,
        total_epochs, lr,
        weight_decay=0, l1_reg=0, early_stopping=-1,
        scheduler_and_args=None, warmup_epochs=0,
        gradient_accumulation=1, optimizer_type=None,
        criterion="mse",
        batch_cosine=False, checkpoints_name=None,
        verbose=False,
        gradient_clipping=None,
        pbar_momentum=0.95,
        device=None,
        repeat_val=1
    ):
    assert criterion in ["mse", "l1"], "Invalid criterion"
    if scheduler_and_args is None or scheduler_and_args[0] != CosineAnnealingLR:
        assert not batch_cosine, "Batch cosine can only be used with CosineAnnealingLR"

    if device is None:
        device = DEVICE

    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    reset_all_seeds()

    if criterion == "mse":
        criterion = nn.MSELoss()
    elif criterion == "l1":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Invalid criterion")

    model.to(device)
    criterion.to(device)

    if optimizer_type is None:
        optimizer_type = optim.Adam
    optimizer = optimizer_type(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_and_args is None:
        main_scheduler = StepLR(optimizer, step_size=2*total_epochs, gamma=1.0)
    else:
        main_scheduler = scheduler_and_args[0](optimizer, **scheduler_and_args[1])
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = main_scheduler

    trajectories = defaultdict(list)

    # Training loop
    with tqdm_nb(total=total_epochs, desc=f"Training Progress", position=0, leave=True, smoothing=0) as pbar0:
        with tqdm_nb(total=len(train_loader), desc="Epoch Progress", position=1, leave=True, smoothing=0) as pbar1:
            best_epoch, best_loss = 0, float("inf")
            running_best_state_dict = None
            epochs_without_improvement = 0

            for epoch in range(total_epochs):
                running_loss = 0
                model.train()
                pbar1.reset(total=len(train_loader))
                optimizer.zero_grad()
                epoch_loss = 0
                for batch_idx, batch_data in enumerate(train_loader):
                    for i, d in enumerate(batch_data):
                        batch_data[i] = d.to(device)
                    label = batch_data[-1].to(device)
                    out = model(batch_data[0], *batch_data[1:-1]).squeeze()

                    loss = criterion(out, label)

                    epoch_loss += loss.item()
                    running_loss = pbar_momentum * running_loss + (1 - pbar_momentum) * loss.item()
                    pbar1.set_postfix({'Loss': f"{running_loss:.4e}"})
                    pbar1.update(1)
                    loss.backward()

                    if gradient_clipping is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                    trajectories["batch_train_loss"].append(loss.item())

                    if l1_reg > 0:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                param.add_(torch.sign(param), alpha=-l1_reg*lr)

                    if (batch_idx + 1) % gradient_accumulation == 0 or batch_idx == len(train_loader) - 1:  # Also check for the last batch
                        optimizer.step()
                        optimizer.zero_grad()

                    if batch_cosine:
                        scheduler.step()

                epoch_loss /= len(train_loader)
                trajectories["epoch_train_loss"].append(epoch_loss)

                # Validation loss
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for _ in range(repeat_val):
                        for batch_data in val_loader:
                            for i, d in enumerate(batch_data):
                                batch_data[i] = d.to(device)
                            label = batch_data[-1].to(device)
                            out = model(batch_data[0], *batch_data[1:-1]).squeeze()
                            loss = criterion(out, label)
                            val_loss += loss.item()
                            trajectories["batch_val_loss"].append(loss.item())

                val_loss /= len(val_loader)*repeat_val
                trajectories["epoch_val_loss"].append(val_loss)

                trajectories["lr"].append(optimizer.param_groups[0]['lr'])

                if (val_loss < best_loss):
                    best_loss = val_loss
                    best_epoch = epoch
                    running_best_state_dict = model.state_dict().copy()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    if not batch_cosine:
                        scheduler.step()

                pbar0.set_postfix({f'Best loss': f"{best_loss:.4f}", 'Best Epoch': f"{best_epoch+1}"})
                pbar0.update(1)
                str_pad = len(str(total_epochs))
                logger.info(f'Epoch [{epoch+1:0>{str_pad}}/{total_epochs}] - Training Loss: {epoch_loss:.4f} - Validation loss: {val_loss:.4f} - Best loss: {best_loss:.4f} (Epoch {best_epoch+1:0>{str_pad}})')
                if verbose:
                    print(f'Epoch [{epoch+1:0>{str_pad}}/{total_epochs}] - Training Loss: {epoch_loss:.4f} - Validation loss: {val_loss:.4f} - Best loss: {best_loss:.4f} (Epoch {best_epoch+1:0>{str_pad}})')

                if checkpoints_name is not None:
                    torch.save(model.state_dict(), f"{PROJECT_FOLDER}/checkpoints/{checkpoints_name}_{epoch+1}.pth")

                if early_stopping > 0 and epochs_without_improvement >= early_stopping:
                    break

    model.load_state_dict(running_best_state_dict)

    del optimizer, criterion, scheduler, label, out, loss, batch_data
    model.cpu()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    for k, v in trajectories.items():
        trajectories[k] = np.array(v)

    return model, trajectories


if __name__ == "__main__":
    # Make sure all pickles are generated
    _get_mrn_csn_map()
    _get_train_tune_test_mrns()
    print("All pickles generated")
    mrn_counts = [len(mrn_to_csns) for mrn_to_csns in [get_train_mrn_csn_map(), get_tune_mrn_csn_map(), get_test_mrn_csn_map()]]
    csn_counts = [sum(len(csn_list) for csn_list in mrn_to_csns.values()) for mrn_to_csns in [get_train_mrn_csn_map(), get_tune_mrn_csn_map(), get_test_mrn_csn_map()]]

    for name, mc, cc in zip(["Train", "Tune", "Test"], mrn_counts, csn_counts):
        print(f"{name}: {mc} MRNs, {cc} CSNs")

    #hours_counts = []
    #for mrn_to_csns in [get_train_mrn_csn_map(), get_tune_mrn_csn_map(), get_test_mrn_csn_map()]:
    #    count = 0
    #    for mrn, csn_list in mrn_to_csns.items():
    #        count += sum(count_hours(mrn, csn) for csn in csn_list)
    #    hours_counts.append(count)


    #for name, mc, cc, hc in zip(["Train", "Tune", "Test"], mrn_counts, csn_counts, hours_counts):
    #    print(f"{name}: {mc} MRNs, {cc} CSNs, {hc:.1f} hours")

    # Create the dataset indices
    # create_dataset_index(get_train_mrn_csn_map(), rmse_lt_3_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10, "train_dataset_index_rmse_lt_3_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10.pkl")
    # create_dataset_index(get_tune_mrn_csn_map(), rmse_lt_3_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10, "tune_dataset_index_rmse_lt_3_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10.pkl")
    # create_dataset_index(get_test_mrn_csn_map(), rmse_lt_3_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10, "test_dataset_index_rmse_lt_3_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10.pkl")

    # create_dataset_index(get_train_mrn_csn_map(), rmse_lt_1_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10, "train_dataset_index_rmse_lt_1_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10.pkl")
    # create_dataset_index(get_tune_mrn_csn_map(), rmse_lt_1_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10, "tune_dataset_index_rmse_lt_1_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10.pkl")
    # create_dataset_index(get_test_mrn_csn_map(), rmse_lt_1_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10, "test_dataset_index_rmse_lt_1_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10.pkl")

    # create_dataset_index(get_train_mrn_csn_map(), r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10, "train_dataset_index_r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10.pkl")
    # create_dataset_index(get_tune_mrn_csn_map(), r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10, "tune_dataset_index_r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10.pkl")
    # create_dataset_index(get_test_mrn_csn_map(), r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10, "test_dataset_index_r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10.pkl")

    # create_dataset_index(get_train_mrn_csn_map(), r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120, "train_dataset_index_r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120.pkl")
    # create_dataset_index(get_tune_mrn_csn_map(), r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120, "tune_dataset_index_r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120.pkl")
    # create_dataset_index(get_test_mrn_csn_map(), r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120, "test_dataset_index_r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120.pkl")

    create_dataset_index(get_train_mrn_csn_map(), r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120_pcrit_lt_100, "train_dataset_index_r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120_pcrit_lt_100.pkl")
    create_dataset_index(get_tune_mrn_csn_map(), r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120_pcrit_lt_100, "tune_dataset_index_r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120_pcrit_lt_100.pkl")
    create_dataset_index(get_test_mrn_csn_map(), r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120_pcrit_lt_100, "test_dataset_index_r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120_pcrit_lt_100.pkl")

    """
    All pickles generated
    Train: 22680 MRNs, 24608 CSNs, 1059681.4 hours
    Tune: 2835 MRNs, 3051 CSNs, 127850.7 hours
    Test: 2835 MRNs, 3076 CSNs, 137349.4 hours

    All pickles generated
    Train: 20774 MRNs, 22336 CSNs
    Tune: 2597 MRNs, 2796 CSNs
    Test: 2597 MRNs, 2815 CSNs
    """