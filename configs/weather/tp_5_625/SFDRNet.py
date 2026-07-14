"""Configuration for SFDRNet on WeatherBench total precipitation at 5.625°."""

# Dataset
levels = []
data_name = "tp"

# OpenSTL method entry
# SFDRNet is implemented through the SimVP experiment interface.
method = "SimVP"

# Model
model_type = "vit"
spatio_kernel_enc = 3
spatio_kernel_dec = 3
hid_S = 32
hid_T = 256
N_S = 2
N_T = 3
drop = 0.0
drop_path = 0.1

# Training
epoch = 200
batch_size = 16
val_batch_size = 16
num_workers = 4
seed = 42
use_augment = False

# Optimization
opt = "adam"
lr = 1e-3
weight_decay = 0.0
sched = "cosine"
warmup_epoch = 0
warmup_lr = 1e-5
min_lr = 1e-6
decay_epoch = 100
decay_rate = 0.1
