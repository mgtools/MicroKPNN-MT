from tqdm import tqdm

import torch
import torch.nn as nn

from captum.attr import (
	IntegratedGradients,
	LayerConductance,
)
import warnings
warnings.filterwarnings("ignore")

from dataset import MicroDataset
from model import MicroKPNN_MTL



def explain_meta_to_disease(model, wrapped_model4disease, batch, device):
	spec, age, gender, bmi, bodysite, disease = [x.type(torch.cuda.FloatTensor).to(device) for x in batch[1:]]
	age_disease_explanations = {}
	gender_disease_explanations = {}
	bmi_disease_explanations = {}
	bodysite_disease_explanations = {}

	# Convert one-hot encoded disease tensor to disease index tensor
	disease_indices = torch.argmax(disease, dim=1)

	# Pred meta-data
	with torch.no_grad():
		pred_age, pred_gender, pred_bmi, pred_bodysite, pred_disease = model(spec, age, gender, bmi, bodysite)
	# Using pred meta-data if the actual is nan
	age = torch.where(torch.isnan(age), pred_age, age)
	gender = torch.where(torch.isnan(gender), pred_gender, gender)
	bmi = torch.where(torch.isnan(bmi), pred_bmi, bmi)
	bodysite = torch.where(torch.isnan(bodysite), pred_bodysite, bodysite)
	for i in range(disease.size(1)): 
		disease_name = f"disease_{i+1}"

		# Create a mask for the current disease index
		disease_mask = (disease_indices == i)

		# Filter the data for the current disease
		filtered_spec = spec[disease_mask].view(-1, spec.size(1))
		filtered_age = age[disease_mask].view(-1, age.size(1))
		filtered_gender = gender[disease_mask].view(-1, gender.size(1))
		filtered_bmi = bmi[disease_mask].view(-1, bmi.size(1))
		filtered_bodysite = bodysite[disease_mask].view(-1, bodysite.size(1))

		# Perform attribution if there are samples for the current disease
		if filtered_spec.size(0) != 0: 
			ig_meta_disease = IntegratedGradients(wrapped_model4disease)
			att, delta = ig_meta_disease.attribute((filtered_spec, filtered_age, filtered_gender, filtered_bmi, filtered_bodysite),
												   baselines=(0, 0, 0, 0, 0),
												   target=i, 
												   n_steps=10, 
												   internal_batch_size=1, 
												   return_convergence_delta=True)

			# Concatenate the attribute tensors of different meta-data
			age_disease_explanations[disease_name] = att[1].detach().cpu()
			gender_disease_explanations[disease_name] = att[2].detach().cpu()
			bmi_disease_explanations[disease_name] = att[3].detach().cpu()
			bodysite_disease_explanations[disease_name] = att[4].detach().cpu()

	return age_disease_explanations, gender_disease_explanations, bmi_disease_explanations, bodysite_disease_explanations

def explain_hidden_to_meta(wrapped_model, model, batch, device, meta_type):
	spec, age, gender, bmi, bodysite = [x.type(torch.cuda.FloatTensor).to(device) for x in batch[1:6]]
	hidden_meta_explanations = {}
	meta_data = {'age': age, 'gender': gender, 'bmi': bmi, 'bodysite': bodysite}
	
	# Convert one-hot encoded meta-data tensor to meta-data index tensor
	meta_indices = torch.argmax(meta_data[meta_type], dim=1)

	for i in range(meta_data[meta_type].size(1)): 
		meta_name = f"{meta_type}_{i+1}"

		# Create a mask for the current disease index
		meta_mask = (meta_indices == i)

		# Filter the data for the current disease
		filtered_spec = spec[meta_mask].view(-1, spec.size(1))
		filtered_age = age[meta_mask].view(-1, age.size(1))
		filtered_gender = gender[meta_mask].view(-1, gender.size(1))
		filtered_bmi = bmi[meta_mask].view(-1, bmi.size(1))
		filtered_bodysite = bodysite[meta_mask].view(-1, bodysite.size(1))
		assert filtered_spec.size(0) == filtered_age.size(0) == filtered_gender.size(0) == filtered_bmi.size(0) == filtered_bodysite.size(0)

		# Perform attribution if there are samples for the current disease
		if filtered_spec.size(0) != 0: 
			lc_hidden_disease = LayerConductance(wrapped_model, model.customized_linear)
			att, delta = lc_hidden_disease.attribute((filtered_spec, filtered_age, filtered_gender, filtered_bmi, filtered_bodysite),
												baselines=(0, 0, 0, 0, 0),  
												target=i, 
												n_steps=10, 
												internal_batch_size=1, 
												return_convergence_delta=True)
			hidden_meta_explanations[meta_name] = att.detach().cpu()
			
	return hidden_meta_explanations

def explain_batch(model, batch, device): 
	# functions from captum only support single output of model, 
	# so we need a wrapper making the model only have one output
	def wrapped_model4disease(spec, age, gender, bmi, bodysite): 
		return model(spec, age, gender, bmi, bodysite)[4]
	def wrapped_model4age(spec, age, gender, bmi, bodysite): 
		return model(spec, age, gender, bmi, bodysite)[0]
	def wrapped_model4gender(spec, age, gender, bmi, bodysite): 
		return model(spec, age, gender, bmi, bodysite)[1]
	def wrapped_model4bmi(spec, age, gender, bmi, bodysite): 
		return model(spec, age, gender, bmi, bodysite)[2]
	def wrapped_model4bodysite(spec, age, gender, bmi, bodysite): 
		return model(spec, age, gender, bmi, bodysite)[3]

	# Move the batch to GPU
	spec, age, gender, bmi, bodysite, disease = [x.type(torch.cuda.FloatTensor).to(device) for x in batch[1:]]

	# Compute explanations for meta-data to disease
	age_disease_explanations, gender_disease_explanations, bmi_disease_explanations, bodysite_disease_explanations = explain_meta_to_disease(model, wrapped_model4disease, batch, device)

	# Compute explanations for hidden layers to each meta-data
	hidden_age_explanations = explain_hidden_to_meta(wrapped_model4age, model, batch, device, 'age')
	hidden_gender_explanations = explain_hidden_to_meta(wrapped_model4gender, model, batch, device, 'gender')
	hidden_bmi_explanations = explain_hidden_to_meta(wrapped_model4bmi, model, batch, device, 'bmi')
	hidden_bodysite_explanations = explain_hidden_to_meta(wrapped_model4bodysite, model, batch, device, 'bodysite')

	# Aggregate results and move them to CPU
	results = {
		'age_disease_explanations': {k: v for k, v in age_disease_explanations.items()}, 
		'gender_disease_explanations': {k: v for k, v in gender_disease_explanations.items()}, 
		'bmi_disease_explanations': {k: v for k, v in bmi_disease_explanations.items()}, 
		'bodysite_disease_explanations': {k: v for k, v in bodysite_disease_explanations.items()}, 

		'hidden_age_explanations': {k: v for k, v in hidden_age_explanations.items()}, 
		'hidden_gender_explanations': {k: v for k, v in hidden_gender_explanations.items()}, 
		'hidden_bmi_explanations': {k: v for k, v in hidden_bmi_explanations.items()}, 
		'hidden_bodysite_explanations': {k: v for k, v in hidden_bodysite_explanations.items()},
	}

	# Clear GPU cache
	torch.cuda.empty_cache()
	return results