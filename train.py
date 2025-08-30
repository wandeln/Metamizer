import matplotlib.pyplot as plt
#from setups_multistep_1_channel import Dataset
from dataset_cloth import DatasetCloth
from dataset_poisson import DatasetPoisson
from dataset_fluid import DatasetFluid
from dataset_diffusion import DatasetDiffusion
from dataset_utils import DatasetToSingleChannel, DatasetConcat
from metamizer import get_Net
#from loss_terms import L_stiffness,L_shearing,L_bending,L_a_ext,L_inertia
from Logger import Logger
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam,get_load_hyperparam,toDType
from utils import has_nan

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)
torch.set_float32_matmul_precision('high')

print(f"Parameters: {vars(params)}")

metamizer = toDType(toCuda(get_Net(params)))
metamizer.train()
#metamizer.nn = torch.compile(metamizer.nn)

steps_per_log = 10

optimizer = AdamW(metamizer.parameters(),lr=params.lr)
scheduler = MultiStepLR(optimizer, milestones=[25,50,75], gamma=0.5)


logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=params.log)
if params.load_latest or params.load_date_time is not None or params.load_index is not None:
	load_logger = Logger(get_load_hyperparam(params),use_csv=False,use_tensorboard=False)
	if params.load_optimizer:
		params.load_date_time, params.load_index = load_logger.load_state(metamizer,optimizer,params.load_date_time,params.load_index)
	else:
		params.load_date_time, params.load_index = load_logger.load_state(metamizer,None,params.load_date_time,params.load_index)
	params.load_index=int(params.load_index)
	print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index

datasets = []
names = []


for iterations_per_timestep in [1,3,10,30]:
#for iterations_per_timestep in [5,25]:
	
	# cloth dataset
	original_dataset_cloth = DatasetCloth(params.height,params.width,params.batch_size,params.dataset_size,params.average_sequence_length,iterations_per_timestep=iterations_per_timestep,stiffness_range=params.stiffness_range,shearing_range=params.shearing_range,bending_range=params.bending_range,a_ext_range=params.g)
	
	#dataset = original_dataset
	dataset_cloth = DatasetToSingleChannel(original_dataset_cloth)
	datasets.append(dataset_cloth)
	names.append(f"cloth_{iterations_per_timestep}")

	# fluid dataset
	original_dataset_fluid = DatasetFluid(params.height,params.width,params.batch_size,params.dataset_size,params.average_sequence_length,iterations_per_timestep=iterations_per_timestep)
	dataset_fluid = DatasetToSingleChannel(original_dataset_fluid)
	datasets.append(dataset_fluid)
	names.append(f"fluid_{iterations_per_timestep}")
	
	# diffusion dataset
	original_dataset_diffusion = DatasetDiffusion(params.height,params.width,params.batch_size,params.dataset_size,average_sequence_length=200,iterations_per_timestep=iterations_per_timestep)
	dataset_diffusion = DatasetToSingleChannel(original_dataset_diffusion)
	datasets.append(dataset_diffusion)
	names.append(f"diffusion_{iterations_per_timestep}")


# poisson dataset
#dataset_poisson = DatasetPoisson(params.height,params.width,params.batch_size*2,params.dataset_size,average_sequence_length=60)
dataset_poisson = DatasetPoisson(params.height,params.width,params.batch_size,params.dataset_size,average_sequence_length=60)
datasets.append(dataset_poisson)
names.append(f"laplace")

dataset = DatasetConcat(datasets,logger,names,steps_per_log)
#dataset = DatasetConcat([dataset_cloth,dataset_poisson,dataset_fluid])
#dataset = DatasetConcat([dataset_cloth,dataset_poisson])
#dataset = DatasetConcat([dataset_fluid])

for epoch in range(int(params.load_index),params.n_epochs):
	print(f"epoch: {epoch} / {params.n_epochs}")
	
	for step in range(params.n_batches_per_epoch):
		
		grads, hidden_states = dataset.ask()
		if has_nan(grads):
			print("input grads contain nan!")
			exit()
		for hs in hidden_states:
			if hs is not None:
				for h in hs:
					if has_nan(h):
						print("input hidden_states contain nan!")
						exit()
		
		update_steps, new_hidden_states = metamizer(grads, hidden_states)
		
		if has_nan(update_steps):
			print("output update_steps contain nan!")
			exit()
		for hs in new_hidden_states:
			for h in hs:
				if has_nan(h):
					print("output hidden_states contain nan!")
					exit()
		
		loss = dataset.tell(update_steps, new_hidden_states)
		
		if step%steps_per_log == 0:
			logger.log(f"L",loss,epoch*params.n_batches_per_epoch+step)
		print(f"({step} / {params.n_batches_per_epoch}): L: {loss}")
		
		if has_nan(loss):
			print(f"loss has nan!")
		
		optimizer.zero_grad()
		loss.backward()
		
		# optional: clip gradients
		if params.clip_grad_value is not None:
			torch.nn.utils.clip_grad_value_(metamizer.parameters(),params.clip_grad_value)
		if params.clip_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(metamizer.parameters(),params.clip_grad_norm)
		
		optimizer.step()
	
	# save state
	if params.log:
		logger.save_state(metamizer.cpu(),optimizer,epoch+1)
		metamizer = toCuda(metamizer)
	
	scheduler.step()
	
# beispiel command:
# python train_multistep.py --log=f --net=Grad_net_tiny --cuda=f --batch_size=1 --dataset_size=1 --n_batches_per_epoch=1000
# python train_multistep.py --log=f --net=Grad_net_scale_inv --cuda=f --batch_size=10 --dataset_size=100 --n_batches_per_epoch=1000 --plot=t --iterations_per_timestep=10
# python train_multistep.py --log=f --net=Grad_net_scale_inv --cuda=f --batch_size=10 --dataset_size=100 --n_batches_per_epoch=1000 --plot=t --iterations_per_timestep=10
# python train_multistep.py --net=Grad_net_scale_inv --batch_size=10 --dataset_size=100 --n_batches_per_epoch=1000 --iterations_per_timestep=10 --cuda=f



