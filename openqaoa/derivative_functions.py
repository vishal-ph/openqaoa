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

"""
Collection of functions to return derivative computation functions. Usually called from the `derivative_function` method of a `QAOABaseBackend` object.
New gradient/higher-order derivative computation methods can be added here. To add new computation methods:
    1. Write function in the format : new_function(backend_obj, params_std, params_ext, gradient_options), or with less arguments.
    2. Give this function a string identifier (eg: 'param_shift'), and add this to the list `derivative_methods` of the function `derivative`, and as a possible 'out'.

"""
import numpy as np
import random


def update_and_compute_expectation(backend_obj, params):
    """
    Helper function that returns a callable that takes in a raw parameters (a list of floats) and returns an expectation value (a float).
    This function will handle: 
        (1) updating variational parameters with QAOAVariationalBaseParams.update_from_raw, and 
        (2) computing expectation with QAOABaseBackend.expectation.
        
    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 
    
    params : QAOAVariationalBaseParams
        `QAOAVariationalBaseParams` object containing variational angles.
    
    """
    
    def fun(args):
        params.update_from_raw(args)
        return backend_obj.expectation(params)

    return fun


def derivative(derivative_dict: dict):
    """
    Called from `cost_function.py`. Returns a callable derivative function, generated according to parameters in `derivative_dict`. 

    PARAMETERS
    ----------
    derivative_dict :
        derivative_type : str
            Type of derivative to compute. String of either `gradient` or `hessian`.

        derivative_method : str
            Computational method of the derivative. String of either `finite_difference`, `param_shift`, `stoch_param_shift`, or `grad_spsa`.

        derivative_options : dict
            Dictionary containing options specific to each `derivative_method` - see their docstrings for options.

        backend_obj : QAOABaseBackend
            `QAOABaseBackend` object that contains information about the abstract circuit to be executed.

        params : QAOAVariationalBaseParams
            `QAOAVariationalBaseParams` object containing variational angles.

        params_ext : VariationalExtendedParams
            extended parametrisation object (for parameter shift and related methods)

    Returns
    -------
    out:
        Callable derivative function that accepts parameters (a list of floats) and returns the gradient (a list of floats).
    """

    derivative_type = derivative_dict['derivative_type']
    derivative_method = derivative_dict['derivative_method']

    # Default derivative_options used if none are specified.
    default_derivative_options = {"stepsize": 0.00001,
                                  "n_beta_single": -1,
                                  "n_beta_pair": -1,
                                  "n_gamma_single": -1,
                                  "n_gamma_pair": -1}

    derivative_options = {**default_derivative_options, **derivative_dict['derivative_options']
                          } if derivative_dict['derivative_options'] is not None else default_derivative_options

    backend_obj = derivative_dict['backend_obj']
    params = derivative_dict['params']
    params_ext = derivative_dict['params_ext']

    derivative_types = ['gradient', 'hessian']
    assert derivative_type in derivative_types,\
        "Unknown derivative type specified - please choose between " + \
        str(derivative_types)

    derivative_methods = ['finite_difference',
                          'param_shift', 'stoch_param_shift', 'grad_spsa']
    assert derivative_method in derivative_methods,\
        "Unknown derivative computation method specified - please choose between " + \
        str(derivative_methods)

    if derivative_type == 'gradient':

        if derivative_method == 'finite_difference':
            out = grad_fd(backend_obj, params, derivative_options)
            
        elif derivative_method == 'param_shift':
            assert params.__class__.__name__ == 'QAOAVariationalStandardParams', f"{params.__class__.__name__} not supported - only Standard Parametrisation is supported for parameter shift/stochastic parameter shift for now."
            out = grad_ps(backend_obj, params, params_ext)
            
        elif derivative_method == 'stoch_param_shift':
            assert params.__class__.__name__ == 'QAOAVariationalStandardParams', f"{params.__class__.__name__} not supported - only Standard Parametrisation is supported for parameter shift/stochastic parameter shift for now."
            out = grad_sps(backend_obj, params, params_ext, derivative_options)
            
        elif derivative_method == 'grad_spsa':
            out = grad_spsa(backend_obj, params, derivative_options)

    elif derivative_type == 'hessian':

        if derivative_method == 'finite_difference':
            out = hessian_fd(backend_obj, params, derivative_options)

    return out


def grad_fd(backend_obj, params, gradient_options):
    """
    Returns a callable function that calculates the gradient with the finite difference method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 

    gradient_options : `dict`
        stepsize : 
            Stepsize of finite difference.

    RETURNS
    -------
    grad_fd_func: `Callable`
        Callable derivative function.
    """

    # Set default value of eta
    eta = gradient_options['stepsize']
    fun = update_and_compute_expectation(backend_obj, params)

    def grad_fd_func(args):

        grad = np.zeros(len(args))

        for i in range(len(args)):
            vect_eta = np.zeros(len(args))
            vect_eta[i] = 1

            # Finite diff. calculation of gradient
            eval_i = fun(args - (eta/2)*vect_eta)
            eval_f = fun(args + (eta/2)*vect_eta)
            grad[i] = (eval_f-eval_i)/eta

        return grad

    return grad_fd_func


def grad_ps(backend_obj, params, params_ext):
    """
    Returns a callable function that calculates the gradient with the parameter shift method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 

    params : `QAOAVariationalStandardParams`
        variational parameters object, standard parametrisation.

    params_ext : `QAOAVariationalExtendedParams`
        variational parameters object, extended parametrisation.

    RETURNS
    -------
    grad_ps_func:
        Callable derivative function.
    """
    
    # TODO : clean up conversion part + handle Fourier parametrisation
    
    fun = update_and_compute_expectation(backend_obj, params_ext)
    
    coeffs_list = params.p*params.mixer_1q_coeffs + params.p*params.mixer_2q_coeffs + \
            params.p*params.cost_1q_coeffs + params.p*params.cost_2q_coeffs
    
    def grad_ps_func(args):

        # Convert standard to extended parameters before applying parameter shift
        args_ext = params.convert_to_ext(args)
        
        grad_ext = np.zeros(len(args_ext))

        # Apply parameter shifts
        for i in range(len(args_ext)):
            vect_eta = np.zeros(len(args_ext))
            vect_eta[i] = 1
            r = coeffs_list[i]
            grad_ext[i] = r*(fun(args_ext + (np.pi/(4*r))*vect_eta) -
                             fun(args_ext - (np.pi/(4*r))*vect_eta))

        # Convert extended param. gradient form back into std param. form
        
        m1q_entries = grad_ext[:params.p*len(params.mixer_1q_coeffs)]
        m2q_entries = grad_ext[params.p*len(params.mixer_1q_coeffs) : params.p*len(params.mixer_2q_coeffs)]
        c1q_entries = grad_ext[params.p*len(params.mixer_1q_coeffs) + params.p*len(params.mixer_2q_coeffs): params.p*len(params.mixer_1q_coeffs) + params.p*len(params.mixer_2q_coeffs) + params.p*len(params.cost_1q_coeffs)]
        c2q_entries = grad_ext[params.p*len(params.mixer_1q_coeffs) + params.p*len(params.mixer_2q_coeffs) + params.p*len(params.cost_1q_coeffs):]
        subdivided_ext_grad = [m1q_entries, m2q_entries, c1q_entries, c2q_entries]
        
        # Sum up gradients (due to the chain rule), and re-express in standard form.
        mat = np.zeros((4, params.p))
        for i in range(4):  # 4 types of terms
            for j in range(params.p):
                mat[i][j] = np.sum(subdivided_ext_grad[i][j*int(len(subdivided_ext_grad[i]) /
                                   params.p):(j+1)*int(len(subdivided_ext_grad[i])/params.p)])

        grad_std = list(np.sum(mat[:2], axis=0)) + \
            list(np.sum(mat[2:], axis=0))
        
        return np.array(grad_std)

    return grad_ps_func


def grad_sps(backend_obj, params_std, params_ext, gradient_options):
    """
    Returns a callable function that approximates the gradient with the stochastic parameter shift method, which samples (n_beta_single, n_beta_pair, n_gamma_single, n_gamma_pair) gates at each layer instead of all gates. See "Algorithm 4" of https://arxiv.org/pdf/1910.01155.pdf. By convention, (n_beta_single, n_beta_pair, n_gamma_single, n_gamma_pair) = (-1, -1, -1, -1) will sample all gates (which is then equivalent to the full parameter shift rule).

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 

    params_std : `QAOAVariationalStandardParams`
        variational parameters object, standard parametrisation.

    params_ext : `QAOAVariationalExtendedParams`
        variational parameters object, extended parametrisation.

    gradient_options :
        n_beta_single : 
            Number of single-qubit mixer gates to sample for the stochastic parameter shift.
        n_beta_pair : 
            Number of X two-qubit mixer gates to sample for the stochastic parameter shift.
        n_gamma_pair : 
            Number of two-qubit cost gates to sample for the stochastic parameter shift.
        n_gamma_single : 
            Number of single-qubit cost gates to sample for the stochastic parameter shift.

    RETURNS
    -------
    grad_sps_func:
        Callable derivative function.
    """
    
    n_beta_single = gradient_options['n_beta_single']
    n_beta_pair = gradient_options['n_beta_pair']
    n_gamma_single = gradient_options['n_gamma_single']
    n_gamma_pair = gradient_options['n_gamma_pair']
    
    beta_single_len = len(params_ext.betas_singles[0])
    beta_pair_len = len(params_ext.betas_pairs[0])
    gamma_single_len = len(params_ext.gammas_singles[0])
    gamma_pair_len = len(params_ext.gammas_pairs[0])
    
    coeffs_list = params_std.p*params_std.mixer_1q_coeffs + params_std.p*params_std.mixer_2q_coeffs + \
            params_std.p*params_std.cost_1q_coeffs + params_std.p*params_std.cost_2q_coeffs
    
    assert (-1 <= n_beta_single <= beta_single_len) and (-1 <= n_beta_pair <= beta_pair_len) and (-1 <= n_gamma_single <= gamma_single_len) and (-1 <= n_gamma_pair <= gamma_pair_len),\
        f"Invalid (n_beta_single, n_beta_pair, n_gamma_pair, n_gamma_single). Each n must be -1 or integers less than or equals to ({str(beta_single_len)}, {str(beta_pair_len)}, {str(gamma_single_len)}, {str(gamma_pair_len)})"

    if n_beta_single == -1: n_beta_single = beta_single_len
    if n_beta_pair == -1: n_beta_pair = beta_pair_len
    if n_gamma_single == -1: n_gamma_single = gamma_single_len
    if n_gamma_pair == -1: n_gamma_pair = gamma_pair_len
        
    fun = update_and_compute_expectation(backend_obj, params_ext)
    
    def grad_sps_func(args):

        # Convert standard to extended parameters before applying parameter shift
        args_ext = params_std.convert_to_ext(args)
        
        grad_ext = np.zeros(len(args_ext))

        # Generate lists of random gates to sample. Note : Gates sampled in each layer is not necessarily the same, but the number of sampled gates in each layer is the same.
        sampled_indices = []
        for p in range(params_std.p):
            sampled_indices.append(random.sample(range(p*len(params_std.mixer_1q_coeffs) , (p+1)*len(params_std.mixer_1q_coeffs)), n_beta_single))

            sampled_indices.append(random.sample(range(params_std.p*len(params_std.mixer_1q_coeffs) + p*len(params_std.mixer_2q_coeffs) , params_std.p*len(params_std.mixer_1q_coeffs) + (p+1)*len(params_std.mixer_2q_coeffs)), n_beta_pair))

            sampled_indices.append(random.sample(range(params_std.p*(len(params_std.mixer_1q_coeffs) + len(params_std.mixer_2q_coeffs)) + p*len(params_std.cost_1q_coeffs) , params_std.p*(len(params_std.mixer_1q_coeffs) + len(params_std.mixer_2q_coeffs)) + (p+1)*len(params_std.cost_1q_coeffs)), n_gamma_single))

            sampled_indices.append(random.sample(range(params_std.p*(len(params_std.mixer_1q_coeffs) + len(params_std.mixer_2q_coeffs) + len(params_std.cost_1q_coeffs)) + p*len(params_std.cost_2q_coeffs) , params_std.p*(len(params_std.mixer_1q_coeffs) + len(params_std.mixer_2q_coeffs) + len(params_std.cost_1q_coeffs)) + (p+1)*len(params_std.cost_2q_coeffs)), n_gamma_pair))
    
        sampled_indices = [item for sublist in sampled_indices for item in sublist]

        # Apply parameter shifts
        for i in range(len(args_ext)):
            if (i) in sampled_indices:
                vect_eta = np.zeros(len(args_ext))
                vect_eta[i] = 1
                r = coeffs_list[i]
                grad_ext[i] = r*(fun(args_ext + (np.pi/(4*r))*vect_eta) -
                                 fun(args_ext - (np.pi/(4*r))*vect_eta))

        # Convert extended param. gradient form back into std param. form
        
        m1q_entries = grad_ext[:params_std.p*len(params_std.mixer_1q_coeffs)]
        m2q_entries = grad_ext[params_std.p*len(params_std.mixer_1q_coeffs) : params_std.p*len(params_std.mixer_2q_coeffs)]
        c1q_entries = grad_ext[params_std.p*len(params_std.mixer_1q_coeffs) + params_std.p*len(params_std.mixer_2q_coeffs): params_std.p*len(params_std.mixer_1q_coeffs) + params_std.p*len(params_std.mixer_2q_coeffs) + params_std.p*len(params_std.cost_1q_coeffs)]
        c2q_entries = grad_ext[params_std.p*len(params_std.mixer_1q_coeffs) + params_std.p*len(params_std.mixer_2q_coeffs) + params_std.p*len(params_std.cost_1q_coeffs):]
        subdivided_ext_grad = [m1q_entries, m2q_entries, c1q_entries, c2q_entries]
        
        # Sum up gradients (due to the chain rule), and re-express in standard form.
        mat = np.zeros((4, params_std.p))
        for i in range(4):  # 4 types of terms
            for j in range(params_std.p):
                mat[i][j] = np.sum(subdivided_ext_grad[i][j*int(len(subdivided_ext_grad[i]) /
                                   params_std.p):(j+1)*int(len(subdivided_ext_grad[i])/params_std.p)])

        grad_std = list(np.sum(mat[:2], axis=0)) + \
            list(np.sum(mat[2:], axis=0))
        
        return np.array(grad_std)

    return grad_sps_func


def hessian_fd(backend_obj, params, hessian_options):
    """
    Returns a callable function that calculates the hessian with the finite difference method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed.

    params : `QAOAVariationalBaseParams`
        variational parameters object.

    hessian_options :
        hessian_stepsize : 
            stepsize of finite difference.

    RETURNS
    -------
    hessian_fd_func:
        Callable derivative function.

    """

    eta = hessian_options['stepsize']
    fun = update_and_compute_expectation(backend_obj, params)

    def hessian_fd_func(args):
        hess = np.zeros((len(args), len(args)))

        for i in range(len(args)):
            for j in range(len(args)):
                vect_eta1 = np.zeros(len(args))
                vect_eta2 = np.zeros(len(args))
                vect_eta1[i] = 1
                vect_eta2[j] = 1

                if i == j:
                    # Central diff. hessian diagonals (https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm)
                    hess[i][i] = (-fun(args+2*eta*vect_eta1) + 16*fun(args + eta*vect_eta1) - 30*fun(
                        args) + 16*fun(args-eta*vect_eta1)-fun(args-2*eta*vect_eta1))/(12*eta**2)
                #grad_diff[i] = (grad_fd_ext(params + (eta/2)*vect_eta)[i] - grad_fd_ext(params - (eta/2)*vect_eta)[i])/eta
                else:
                    hess[i][j] = (fun(args + eta*vect_eta1 + eta*vect_eta2)-fun(args + eta*vect_eta1 - eta*vect_eta2)-fun(
                        args - eta*vect_eta1 + eta*vect_eta2)+fun(args - eta*vect_eta1 - eta*vect_eta2))/(4*eta**2)

        return hess

    return hessian_fd_func


def grad_spsa(backend_obj, params, gradient_options):
    """
    Returns a callable function that calculates the gradient approxmiation with the Simultaneous Perturbation Stochastic Approximation (SPSA) method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 

    params : `QAOAVariationalBaseParams`
        variational parameters object.

    gradient_options : `dict`
        gradient_stepsize : 
            stepsize of stochastic shift.

    RETURNS
    -------
    grad_spsa_func: `Callable`
        Callable derivative function.

    """
    c = gradient_options['stepsize']
    fun = update_and_compute_expectation(backend_obj, params)

    def grad_spsa_func(args):
        delta = (2*np.random.randint(0, 2, size=len(args))-1)
        return np.real((fun(args + c*delta) - fun(args - c*delta))*delta/(2*c))

    return grad_spsa_func
