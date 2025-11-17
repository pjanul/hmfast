"""
Clean JAX-based neural network emulator classes.

This module provides pure JAX implementations without any tensorflow dependencies,
specifically designed for loading cosmopower-style .npz emulator files.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, List, Union
from functools import partial

# Enable 64-bit precision for better accuracy
jax.config.update("jax_enable_x64", True)


class EmulatorLoader:
    """
    Pure JAX neural network restoration class.
    Loads cosmopower-style .npz files without any tensorflow dependencies.
    """
    
    def __init__(self, restore_filename: str):
        """
        Initialize neural network from saved .npz file.
        
        Parameters
        ----------
        restore_filename : str
            Path to the .npz file containing model weights
        """
        self.restore_filename = restore_filename
        self.restore(restore_filename)
        
        # Convert to JAX arrays for efficient computation
        self.parameters_mean = jnp.array(self.parameters_mean_, dtype=jnp.float64)
        self.parameters_std = jnp.array(self.parameters_std_, dtype=jnp.float64)
        self.features_mean = jnp.array(self.features_mean_, dtype=jnp.float64)
        self.features_std = jnp.array(self.features_std_, dtype=jnp.float64)
        
        # Convert weights and biases to JAX arrays
        self.W_ = [jnp.array(w, dtype=jnp.float64) for w in self.W_]
        self.b_ = [jnp.array(b, dtype=jnp.float64) for b in self.b_]
        self.alphas_ = [jnp.array(alpha, dtype=jnp.float64) for alpha in self.alphas_]
        self.betas_ = [jnp.array(beta, dtype=jnp.float64) for beta in self.betas_]
    
    def restore(self, filename: str) -> None:
        """
        Load pre-trained model from .npz file.
        
        Parameters
        ----------
        filename : str
            Path to .npz file (without extension)
        """
        filename_npz = filename + ".npz"
        if not os.path.exists(filename_npz):
            raise IOError(f"Failed to restore network from {filename}: does not exist.")
        
        with open(filename_npz, "rb") as fp:
            fpz = np.load(fp, allow_pickle=True)["arr_0"].flatten()[0]
            
            self.architecture = fpz["architecture"]
            self.n_layers = fpz["n_layers"]
            self.n_hidden = fpz["n_hidden"]
            self.n_parameters = fpz["n_parameters"]
            self.n_modes = fpz["n_modes"]
            
            self.parameters = list(fpz["parameters"])
            self.modes = fpz["modes"]
            
            # Handle different naming conventions
            self.parameters_mean_ = fpz.get("parameters_mean", fpz.get("param_train_mean"))
            self.parameters_std_ = fpz.get("parameters_std", fpz.get("param_train_std"))
            self.features_mean_ = fpz.get("features_mean", fpz.get("feature_train_mean"))
            self.features_std_ = fpz.get("features_std", fpz.get("feature_train_std"))
            
            # Load weights and biases
            if "weights_" in fpz:
                self.W_ = fpz["weights_"]
            else:
                self.W_ = [fpz[f"W_{i}"] for i in range(self.n_layers)]
            
            if "biases_" in fpz:
                self.b_ = fpz["biases_"]
            else:
                self.b_ = [fpz[f"b_{i}"] for i in range(self.n_layers)]
            
            self.alphas_ = fpz["alphas_"]
            self.betas_ = fpz["betas_"]
    
    def dict_to_ordered_arr(self, input_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Sort input parameters according to the expected order.
        
        Parameters
        ----------
        input_dict : dict
            Dictionary of parameters
            
        Returns
        -------
        jnp.ndarray
            Ordered parameter array
        """
        if self.parameters is not None:
            return jnp.stack([input_dict[k] for k in self.parameters], axis=-1)
        else:
            return jnp.stack([input_dict[k] for k in input_dict], axis=-1)
    
    @partial(jax.jit, static_argnums=(0,))
    def forward_pass(self, parameters_arr: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the neural network.
        
        Parameters
        ----------
        parameters_arr : jnp.ndarray
            Input parameters
            
        Returns
        -------
        jnp.ndarray
            Network predictions
        """
        # Normalize inputs
        layers = [(parameters_arr - self.parameters_mean) / self.parameters_std]
        
        # Forward pass through hidden layers
        for i in range(self.n_layers - 1):
            # Linear operation
            act = jnp.dot(layers[-1], self.W_[i]) + self.b_[i]
            
            # Activation function with learned parameters
            activated = (self.betas_[i] + (1.0 - self.betas_[i]) * 
                        1.0 / (1.0 + jnp.exp(-self.alphas_[i] * act))) * act
            layers.append(activated)
        
        # Final layer
        output = jnp.dot(layers[-1], self.W_[-1]) + self.b_[-1]
        
        # Denormalize output
        return output * self.features_std + self.features_mean
    
    def predictions(self, parameters_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Make predictions given input parameters.
        
        Parameters
        ----------
        parameters_dict : dict
            Dictionary of cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Network predictions
        """
        parameters_arr = self.dict_to_ordered_arr(parameters_dict)
        return self.forward_pass(parameters_arr)
    
    def ten_to_predictions(self, parameters_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Return 10^predictions for log-scale outputs.
        
        Parameters
        ----------
        parameters_dict : dict
            Dictionary of cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            10^predictions
        """
        return 10.0 ** self.predictions(parameters_dict)


class EmulatorLoaderPCA:
    """
    Pure JAX neural network with PCA compression.
    No tensorflow dependencies.
    """
    
    def __init__(self, restore_filename: str):
        """
        Initialize PCA+NN model from saved .npz file.
        
        Parameters
        ----------
        restore_filename : str
            Path to the .npz file containing model weights
        """
        self.restore_filename = restore_filename
        self.restore(restore_filename)
        
        # Convert to JAX arrays
        self.parameters_mean = jnp.array(self.parameters_mean_, dtype=jnp.float64)
        self.parameters_std = jnp.array(self.parameters_std_, dtype=jnp.float64)
        self.pca_mean = jnp.array(self.pca_mean_, dtype=jnp.float64)
        self.pca_std = jnp.array(self.pca_std_, dtype=jnp.float64)
        self.features_mean = jnp.array(self.features_mean_, dtype=jnp.float64)
        self.features_std = jnp.array(self.features_std_, dtype=jnp.float64)
        self.pca_transform_matrix = jnp.array(self.pca_transform_matrix_, dtype=jnp.float64)
        
        # Convert weights and biases to JAX arrays
        self.W_ = [jnp.array(w, dtype=jnp.float64) for w in self.W_]
        self.b_ = [jnp.array(b, dtype=jnp.float64) for b in self.b_]
        self.alphas_ = [jnp.array(alpha, dtype=jnp.float64) for alpha in self.alphas_]
        self.betas_ = [jnp.array(beta, dtype=jnp.float64) for beta in self.betas_]
    
    def restore(self, filename: str) -> None:
        """Load pre-trained PCA+NN model from .npz file."""
        filename_npz = filename + ".npz"
        if not os.path.exists(filename_npz):
            raise IOError(f"Failed to restore network from {filename}: does not exist.")
        
        with open(filename_npz, "rb") as fp:
            fpz = np.load(fp, allow_pickle=True)["arr_0"].flatten()[0]
            
            self.architecture = fpz["architecture"]
            self.n_layers = fpz["n_layers"]
            self.n_hidden = fpz["n_hidden"]
            self.n_parameters = fpz["n_parameters"]
            self.n_modes = fpz["n_modes"]
            self.n_pcas = fpz["n_pcas"]
            
            self.parameters = fpz["parameters"]
            self.modes = fpz["modes"]
            
            # Parameters normalization
            self.parameters_mean_ = fpz.get("parameters_mean", fpz.get("param_train_mean"))
            self.parameters_std_ = fpz.get("parameters_std", fpz.get("param_train_std"))
            self.features_mean_ = fpz.get("features_mean", fpz.get("feature_train_mean"))
            self.features_std_ = fpz.get("features_std", fpz.get("feature_train_std"))
            
            # PCA components
            self.pca_mean_ = fpz["pca_mean"]
            self.pca_std_ = fpz["pca_std"]
            self.pca_transform_matrix_ = fpz["pca_transform_matrix"]
            
            # Weights and biases
            if "weights_" in fpz:
                self.W_ = fpz["weights_"]
            else:
                self.W_ = [fpz[f"W_{i}"] for i in range(self.n_layers)]
            
            if "biases_" in fpz:
                self.b_ = fpz["biases_"]
            else:
                self.b_ = [fpz[f"b_{i}"] for i in range(self.n_layers)]
            
            self.alphas_ = fpz.get("alphas_", [fpz.get(f"alphas_{i}") for i in range(self.n_layers - 1)])
            self.betas_ = fpz.get("betas_", [fpz.get(f"betas_{i}") for i in range(self.n_layers - 1)])
    
    def dict_to_ordered_arr(self, input_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """Sort input parameters according to expected order."""
        if self.parameters is not None:
            return jnp.stack([input_dict[k] for k in self.parameters], axis=-1)
        else:
            return jnp.stack([input_dict[k] for k in input_dict], axis=-1)
    
    @partial(jax.jit, static_argnums=(0,))
    def forward_pass(self, parameters_arr: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through PCA+NN network.
        
        Parameters
        ----------
        parameters_arr : jnp.ndarray
            Input parameters
            
        Returns
        -------
        jnp.ndarray
            Network predictions
        """
        # Normalize inputs
        layers = [(parameters_arr - self.parameters_mean) / self.parameters_std]
        
        # Forward pass through hidden layers
        for i in range(self.n_layers - 1):
            # Linear operation
            act = jnp.dot(layers[-1], self.W_[i]) + self.b_[i]
            
            # Activation function
            activated = (self.betas_[i] + (1.0 - self.betas_[i]) * 
                        1.0 / (1.0 + jnp.exp(-self.alphas_[i] * act))) * act
            layers.append(activated)
        
        # Final layer -> PCA coefficients
        pca_coeffs = jnp.dot(layers[-1], self.W_[-1]) + self.b_[-1]
        
        # Reconstruct from PCA: denormalize coefficients, apply PCA transform, denormalize output
        pca_denorm = pca_coeffs * self.pca_std + self.pca_mean
        reconstructed = jnp.dot(pca_denorm, self.pca_transform_matrix)
        output = reconstructed * self.features_std + self.features_mean
        
        return output
    
    def predictions(self, parameters_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """Make predictions given input parameters."""
        parameters_arr = self.dict_to_ordered_arr(parameters_dict)
        return self.forward_pass(parameters_arr)
    
    def ten_to_predictions(self, parameters_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """Return 10^predictions for log-scale outputs."""
        return 10.0 ** self.predictions(parameters_dict)