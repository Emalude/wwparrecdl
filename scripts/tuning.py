import wwparrecdl as pr

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import numpy as np
from scipy.stats import linregress
from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = Path(__file__).parent.parent.absolute() / 'wwparrecdl/data'
model_dir = data_path / 'models'
data_dir = data_path / 'dl_input'
checkpoints_dir = model_dir / 'checkpoints'

config = {
    "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 10)),
    "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 10)),
    "l3": tune.sample_from(lambda _: 2**np.random.randint(2, 10)),
    "lr": tune.loguniform(1e-9, 1e-1),
    "batch_size": tune.choice([25, 50, 75, 100, 125, 150]),
    "mom": tune.loguniform(0.5, 0.9999)
}

max_num_epochs=50
scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)

reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "l3", batch_size"],
    metric_columns=["loss", "R2_1","R2_2", "R2_3", "training_iteration"])

num_samples=100
gpus_per_trial=1
result = tune.run(
    partial(pr.train_tune, data_dir=data_dir),
    resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter)

n_features = 13
n_points   = 1
n_outputs  = 3

prnet = pr.PRNet(n_features*n_points, config['l1'], config['l2'], config['l3'], n_outputs)

if torch.cuda.device_count() > 1:
    prnet = DataParallel(net)

prnet.to(device)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(prnet.parameters(), lr=config['lr'], momentum = config['mom'])

pr.train_tune(prnet, loss_function, optimizer, train_dataloader, test_set, num_epochs=100)
