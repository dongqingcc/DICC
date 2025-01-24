# training config
n_step = 1000000
scheduler_checkpoint_step = 800
log_checkpoint_step = 25
gradient_accumulation_steps=1

lr = 3e-5
batch_size = 4
val_batch_size = 64

decay = 0.9
minf = 0.5
optimizer = "adam"  # adamw or adam
n_workers = 4

# load
load_model = False
load_step = False

# diffusion config
alpha=0.9
beta=0.0025


pred_mode = 'noise'
loss_type = "l1"
iteration_step =  20000                   #20000
sample_steps = 500     #500
embed_dim = 64    #64
dim_mults = (1, 2, 3, 4, 5, 6)
hyper_dim_mults = (4, 4, 4)
context_channels = 3
clip_noise = "none"
val_num_of_batch = 1
additional_note = ""
vbr = False
context_dim_mults = (1, 2, 3, 4)
sample_mode = "ddim"
var_schedule = "linear"
aux_loss_type = "lpips"
compressor = "big"

# data config
data_config = {
    "dataset_name": "Cityscape",
    "data_path": "/data/code/data/Cityscape",
    "sequence_length": 1,
    "img_size": 256,
    "img_channel": 3,
    "add_noise": False,
    "img_hz_flip": False,
}



result_root = "./city"
tensorboard_root = "*"
