import sys
sys.path = ['/user/sunsiqi/hs/PAD-Net-master/moe', '/user/sunsiqi/hs/PAD-Net-master/moe/petl',
            '/user/sunsiqi/hs/PAD-Net-master/moe/transformers', '/user/sunsiqi/hs/PAD-Net-master/apex'] + \
           sys.path # todo add your own path of transformers if you use local transformer packages.

from apex.contrib.sparsity import ASP
from transformers.models.layers import MoE, SMoE, AMoE, STRMoE, PadMoE, PadMoE_Structured
from transformers import (
    AutoConfig,

)
import time
import torch
from torch import optim
from tqdm import tqdm
from torch.nn import DataParallel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
device = torch.device('cuda:0')
input_size = 3072
hidden_size = 768
config = AutoConfig.from_pretrained('bert-base-cased')
setattr(config, 'n_experts', 16)
setattr(config, 'k', 4)
setattr(config, 'Lambda', 'none')
setattr(config, 'moe_level', 'token')

# model = DataParallel(PadMoE(input_size, hidden_size, config)).to(device)
model = DataParallel(MoE(input_size, hidden_size, config)).to(device)
# model = PadMoE_Structured(input_size, hidden_size, config).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # Define optimizer
ASP.prune_trained_model(model, optimizer)

max_length, batch_size = 128, 64
steps = 1000
time_start = time.time()
for step in tqdm(range(steps)):
    inputs = torch.randn((batch_size, max_length, input_size), device=device)
    model(inputs)

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
