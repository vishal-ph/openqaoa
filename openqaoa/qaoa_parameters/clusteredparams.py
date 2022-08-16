#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typing import Tuple, List, Union
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from openqaoa.qaoa_parameters.operators import Hamiltonian
from .baseparams import QAOACircuitParams, QAOAVariationalBaseParams, shapedArray, _is_iterable_empty


def hamiltonian_clustering(qaoa_circuit_params: QAOACircuitParams,
                           max_std_dev: float,
                           n_clusters: int):

    if n_clusters > 0 :
        print('Using k-means')
        cost_hamiltonian = qaoa_circuit_params.cost_hamiltonian
        coeffs = cost_hamiltonian.coeffs
        terms = cost_hamiltonian.terms
        sorted_terms,sorted_coeffs = zip(*sorted(zip(terms,coeffs), key = lambda pair: pair[1]))

        X = np.array(sorted(np.array(sorted_coeffs).reshape(-1, 1)))
        kmeans = KMeans(n_clusters=n_clusters).fit(X)

        clusters = []
        for l in range(len(set(kmeans.labels_))):
            clusters.append(Hamiltonian(np.array(sorted_terms)[kmeans.labels_==l],X[kmeans.labels_==l],0,remap_logical_qubits=False))

    else:
        print('Using the sharma parametrisation')
        cost_hamiltonian = qaoa_circuit_params.cost_hamiltonian
        coeffs = cost_hamiltonian.coeffs
        terms = cost_hamiltonian.terms
        sorted_terms,sorted_coeffs = zip(*sorted(zip(terms,coeffs), key = lambda pair: pair[1]))

        sorted_terms = list(sorted_terms)
        sorted_coeffs = list(sorted_coeffs)

        clusters=[]
        new_cluster_terms = []
        new_cluster_coeffs = []
        for term,coeff in zip(sorted_terms,sorted_coeffs):	
            new_cluster_terms.append(term)
            new_cluster_coeffs.append(coeff)
            current_std_dev = np.std(new_cluster_coeffs)
            if current_std_dev >= max_std_dev:
                clusters.append(Hamiltonian(new_cluster_terms,new_cluster_coeffs,0,remap_logical_qubits=False))
                new_cluster_terms = []
                new_cluster_coeffs = []
        clusters.append(Hamiltonian(new_cluster_terms,new_cluster_coeffs,0,remap_logical_qubits=False))	
    
    return clusters


class QAOAVariationalClusteredParams(QAOAVariationalBaseParams):
	"""
	QAOA parameters set in accordance to hamiltonian terms, grouping high magnitude
	terms together with a corresponding parameters

	This means, that at the i-th timestep the evolution hamiltonian is given by

	.. math::

		H(t_i) = \sum_{\textrm{qubits } j} \beta X_j
			   + \sum_{\textrm{qubit terms} (jk)} \gamma_i} Z_j Z_k

	and the complete circuit is then

	.. math::

		U = e^{-i H(t_p)} \cdots e^{-iH(t_1)}.

	Parameters
	----------
	qaoa_circuit_params: QAOACircuitParams
		Specify the circuit parameters to construct circuit angles to be
		used for training
	betas: list
		2D array with the betas from above for each timestep and qubit.
		1st index goes over the timelayers, 2nd over the qubits.
	gammas: list
		2D array with the gammas from above for each timestep and qubit.

	Attributes
	----------
	qaoa_circuit_params: QAOACircuitParams
		Specify the circuit parameters to construct circuit angles to be
		used for training
	betas: list
		2D array with the betas from above for each timestep and qubit.
		1st index goes over the timelayers, 2nd over the qubits.
	gammas: list
		2D array with the gammas from above for each timestep and qubit.
	"""

	def __init__(self,
				 qaoa_circuit_params: QAOACircuitParams,
				 max_std_dev: float,
                 n_clusters: int,
				 betas: List[Union[float, int]],
				 gammas: List[Union[float, int]]):   

		# setup reg, qubits_singles and qubits_pairs
		super().__init__(qaoa_circuit_params)
		self.max_std_dev = max_std_dev
		self.n_clusters = n_clusters
		self.hamiltonian_clusters = hamiltonian_clustering(self.qaoa_circuit_params, self.max_std_dev, self.n_clusters)
		# and add the parameters
		self.betas = betas
		self.gammas = gammas
		self.cost_1q_terms = self.qaoa_circuit_params.cost_hamiltonian.qubits_singles
		self.cost_2q_terms = self.qaoa_circuit_params.cost_hamiltonian.qubits_pairs

		assert self.gammas.shape == (self.p, len(self.hamiltonian_clusters)), f'Please specify ,\
									{(self.p,len(self.hamiltonian_clusters))} unique gammas'

	def __repr__(self):
		string = "Clustering Parameterisation:\n"
		string += "\tp: " + str(self.p) + "\n"
		string += "Parameters:\n"
		string += "\tbetas: " + str(self.betas).replace("\n", ",") + "\n"
		string += "\tgammas: " + str(self.gammas).replace("\n", ",") + "\n"
		return string

	def __len__(self):
		return self.p + len(self.hamiltonian_clusters)

	@shapedArray
	def betas(self):
		return self.p

	@shapedArray
	def gammas(self):
		return (self.p, len(self.hamiltonian_clusters))

	@property
	def mixer_1q_angles(self):
		return 2*np.outer(self.betas, self.mixer_1q_coeffs)

	@property
	def mixer_2q_angles(self):
		return 2*np.outer(self.betas, self.mixer_2q_coeffs)

	@property
	def cost_1q_angles(self):
		rotation_angles=[]
		for p in range(self.p):
			for i,cluster in enumerate(self.hamiltonian_clusters):
				cluster_mean = np.mean(cluster.coeffs)
				rotation_angles.extend([2*self.cost_1q_coeffs[k]*self.gammas[p][i]/cluster_mean 
						for k,term in enumerate(self.cost_1q_terms) if term in cluster.terms])
		rotation_angles = np.array(rotation_angles).reshape(self.p,len(self.cost_1q_coeffs))
		return rotation_angles

	@property
	def cost_2q_angles(self):
		rotation_angles=[]
		for p in range(self.p):
			for i,cluster in enumerate(self.hamiltonian_clusters):
				cluster_mean = np.mean(cluster.coeffs)
				rotation_angles.extend([2*self.cost_2q_coeffs[k]*self.gammas[p][i]/cluster_mean 
					   for k,term in enumerate(self.cost_2q_terms) if term in cluster.terms])
		rotation_angles = np.array(rotation_angles).reshape(self.p,len(self.cost_2q_coeffs))
		return rotation_angles
	
	def update_from_raw(self, new_values):
		self.betas = np.array(new_values[:self.p])
		new_values = new_values[self.p:]

		self.gammas = np.array(new_values)
		self.gammas = self.gammas.reshape((self.p,len(self.hamiltonian_clusters)))
		
		new_values = new_values[self.p * len(self.hamiltonian_clusters):]

		# PEP8 complains, but new_values could be np.array and not list!
		if not len(new_values) == 0:
			raise RuntimeWarning(
				"list to make new gammas and x_rotation_angles out of didn't"
				"have the right length!")

	def raw(self):
		raw_data = np.concatenate((self.betas.flatten(),
								   self.gammas.flatten()))
		return raw_data

	@classmethod
	def linear_ramp_from_hamiltonian(cls,
									 qaoa_circuit_params:QAOACircuitParams,
									 max_std_dev: float,
									 n_clusters: int = 0,
									 time: float = None):
		"""

		Returns
		-------
		ClusteredParams
			The initial parameters according to a linear ramp for the Hamiltonian specified by
			register, terms, weights.

		Todo
		----
		Refactor this s.t. it supers from __init__
		"""
		# create evenly spaced timelayers at the centers of p intervals
		p = qaoa_circuit_params.p
		if time is None:
			time = float(0.7 * p)

		dt = time / p

		hamiltonian_clusters = hamiltonian_clustering(qaoa_circuit_params, max_std_dev, n_clusters)
		n_gammas = len(hamiltonian_clusters)
		betas = np.linspace((dt / time) * (time * (1 - 0.5 / p)),
							(dt / time) * (time * 0.5 / p), p)
		gammas = betas[::-1]        
		gammas = gammas.repeat(n_gammas).reshape(p, n_gammas)

		# wrap it all nicely in a qaoa_parameters object
		params = cls(qaoa_circuit_params, max_std_dev, n_clusters, betas, gammas)
		return params
	
	@classmethod
	def random(cls,qaoa_circuit_params:QAOACircuitParams,max_std_dev:float,n_clusters:int=0,seed:int = None):
		"""
		Returns
		-------
		ClusteredParams
			Randomly initialised ``ClusterdParams`` object
		"""
		if seed is not None:
			np.random.seed(seed)
			
		p = qaoa_circuit_params.p
		hamiltonian_clusters = hamiltonian_clustering(qaoa_circuit_params, max_std_dev, n_clusters)
		n_gammas = len(hamiltonian_clusters)
		betas = np.random.uniform(0,np.pi,p)
		gammas = np.random.uniform(0,np.pi,(p,n_gammas))

		params = cls(qaoa_circuit_params, max_std_dev, n_clusters, betas, gammas)
		return params

	@classmethod
	def empty(cls, qaoa_circuit_params:QAOACircuitParams,max_std_dev: float, n_clusters:int):

		p = qaoa_circuit_params.p
		hamiltonian_clusters = hamiltonian_clustering(qaoa_circuit_params, max_std_dev, n_clusters)
		betas = np.empty(p)
		gammas = np.empty(p, len(hamiltonian_clusters))
		return cls(qaoa_circuit_params, max_std_dev, n_clusters, betas, gammas)

	def get_constraints(self):
		"""Constraints on the parameters for constrained parameters.

		Returns
		-------
		List[Tuple]:
			A list of tuples (0, upper_boundary) of constraints on the
			parameters s.t. we are exploiting the periodicity of the cost
			function. Useful for constrained optimizers.
		"""
		beta_constraints = [(0, 2 * math.pi)] * len(self.betas.flatten())
		gammas_constraints = [(0, 2 * math.pi / w)
								  for w in self.hamiltonian_clusters]
		gammas_constraints *= self.p
		
		all_constraints = beta_constraints + gammas_constraints
		return all_constraints

	def plot(self, ax=None, **kwargs):
		if ax is None:
			fig, ax = plt.subplots()
			
		ax.plot(self.betas, label="betas", marker="s", ls="", **kwargs)
		if not _is_iterable_empty(self.gammas):
			ax.plot(self.gammas,
					label="gammas", marker="^", ls="", **kwargs)
		ax.set_xlabel("timestep")
		ax.legend()