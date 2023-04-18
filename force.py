#!/usr/bin/env python
# coding: utf-8

# # Training a model on forces and energies
# 
# In addition to the energy, machine learning models can also be used to model molecular forces.
# These are $N_\mathrm{atoms} \times 3$ arrays describing the Cartesian force acting on each atom due to the overall
# (potential) energy. They are formally defined as the negative gradient of the energy $E_\mathrm{pot}$ with respect to
# the nuclear positions $\mathbf{R}$
# 
# \begin{equation}
# \mathbf{F}^{(\alpha)} = -\frac{\partial E_\mathrm{pot}}{\partial \mathbf{R}^{(\alpha)}},
# \end{equation}
# 
# where $\alpha$ is the index of the nucleus.
# 
# The above expression offers a straightforward way to include forces in machine learning models by simply defining a
# model for the energy and taking the appropriate derivatives.
# The resulting model can directly be trained on energies and forces.
# Moreover, in this manner energy conservation and the correct behaviour under rotations of the molecule is guaranteed.
# 
# Using forces in addition to energies to construct a machine learning model offers several advantages.
# Accurate force predictions are important for molecular dynamics simulations, which will be covered in the subsequent
# tutorial. Forces also encode a greater wealth of information than the energies.
# For every molecule, only one energy is present, while there are $3N_\mathrm{atoms}$ force entries.
# This property, combined with the fact that reference forces can be computed at the same cost as energies, makes models
# trained on forces and energies very data efficient.
# 
# In the following, we will show how to train such force models and how to use them in practical applications.

# ## Preparing the data
# 
# The process of preparing the data is similar to the tutorial on [QM9](tutorial_02_qm9.ipynb). We begin by importing all
# relevant packages and generating a directory for the tutorial experiments.

# In[1]:


import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np

forcetut = './forcetut'
if not os.path.exists(forcetut):
    os.makedirs(forcetut)


# Next, the data needs to be loaded from a suitable dataset. 
# For convenience, we use the MD17 dataset class provided in SchNetPack, which automatically downloads and builds suitable
# databases containing energies and forces for a range of small organic molecules.
# In this case, we use the ethanol molecule as an example.

# In[2]:


from schnetpack.data import ASEAtomsData,AtomsDataModule



MoS2_1T = AtomsDataModule(
    os.path.join('MoS2_1T.db'),
    batch_size=10,
    num_train=8000,
    num_val=1000,
    num_test=1000,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=1,
    pin_memory=True, # set to false, when not using a GPU
)


MoS2_1T.setup()


cutoff = 5.
n_atom_basis = 64

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=300, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=6,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)

pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='energy')
pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets('energy', add_mean=True, add_atomrefs=False)
    ]
)

output_energy = spk.task.ModelOutput(
    name='energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

output_forces = spk.task.ModelOutput(
    name='forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.99,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-3}
)

logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(forcetut, "best3_inference_model"),
        save_top_k=3,
        monitor="val_loss"
    )
]
pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", min_delta=0.0, patience=5, verbose=False, mode='min',strict=True)
trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=forcetut,
    max_epochs=1000, # for testing, we restrict the number of epochs
)
print(MoS2_1T.train_dataset[0])
trainer.fit(task, datamodule= MoS2_1T)
