import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MOE_ROOT = _SCRIPT_DIR.parent
if str(_MOE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOE_ROOT))

try:
    from apex.contrib.sparsity import ASP
except ImportError:
    ASP = None
from transformers.models.layers import MoE, SMoE, AMoE, STRMoE, PadMoE, PadMoE_Structured
from transformers import (
    AutoConfig,

)
import time
import torch
from torch import optim
from tqdm import tqdm
from torch.nn import DataParallel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 3072
hidden_size = 768
config = AutoConfig.from_pretrained('bert-base-cased')
setattr(config, 'n_experts', 16)
setattr(config, 'k', 4)
setattr(config, 'Lambda', 'none')
setattr(config, 'moe_level', 'token')

# model = DataParallel(PadMoE(input_size, hidden_size, config)).to(device)
if torch.cuda.is_available():
    model = DataParallel(MoE(input_size, hidden_size, config)).to(device)
else:
    model = MoE(input_size, hidden_size, config).to(device)
# model = PadMoE_Structured(input_size, hidden_size, config).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # Define optimizer
if ASP is not None:
    ASP.prune_trained_model(model, optimizer)

max_length, batch_size = 128, 64
steps = 1000
time_start = time.time()
for step in tqdm(range(steps)):
    inputs = torch.randn((batch_size, max_length, input_size), device=device)
    model(inputs)

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
