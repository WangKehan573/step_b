import os

# The data set is in Numpy format. 
# In the following, we show how this data can be parsed and converted for use in SchNetPack, so that you apply this to any other data format.
# 
# First, we need to parse our data. For this we use the IO functionality supplied by ASE.
# In order to create a SchNetPack DB, we require a **list of ASE `Atoms` objects** as well as a corresponding **list of dictionaries** `[{property_name1: property1_molecule1}, {property_name1: property1_molecule2}, ...]` containing the mapping from property names to values.

# In[9]:


from ase import Atoms
import numpy as np
import torch

from schnetpack.data import ASEAtomsData

# load atoms from npz file. Here, we only parse the first 10 molecules
data = np.load('./MoS2_2H.npz',allow_pickle=True)

numbers = data["z"]
atoms_list = []
property_list = []
#cell = torch.Tensor([[19.442784 ,0,0],[-9.721392,16.837944,0],[0,0,36.356154]])

for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
    ats = Atoms(positions=positions, numbers=numbers)
    ats.set_pbc((True,True,True))
    ats.set_cell([[6.39818,0,0],[-3.19909,5.540988 ,0],[0,0,12.42199]])
    properties = {'energy': energies, 'forces': forces}
    property_list.append(properties)
    atoms_list.append(ats)

    
print('Properties:', property_list[0])


# Once we have our data in this format, it is straightforward to create a new SchNetPack DB and store it.

# In[10]:


get_ipython().run_line_magic('rm', "'./MoS2_1T.db'")
new_dataset = ASEAtomsData.create(
    './MoS2_1T.db', 
    distance_unit='Ang',
    property_unit_dict={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'}
)
new_dataset.add_systems(property_list, atoms_list)


# Now we can have a look at the data in the same way we did before for QM9:

# In[11]:


print('Number of reference calculations:', len(new_dataset))
print('Available properties:')

for p in new_dataset.available_properties:
    print('-', p)
   

example = new_dataset[12000]
print('Properties of molecule with id 0:')

for k, v in example.items():
    print('-', k, ':', v)


# The same way, we can store multiple properties, including atomic properties such as forces, or tensorial properties such as polarizability tensors.
# 
# In the following tutorials, we will describe how these datasets can be used to train neural networks.

# In[ ]:
