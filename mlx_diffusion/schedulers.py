import math
from typing import Optional, Union, List
import mlx.core as mx
import numpy as np
from abc import ABC, abstractmethod


class SchedulerMixin(ABC):
    """Base class for all schedulers"""
    
    def __init__(self, 
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.00085,
                 beta_end: float = 0.012,
                 beta_schedule: str = "scaled_linear"):
        
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)
        
        # Add padding for prev cumulative products
        self.alphas_cumprod_prev = mx.concatenate([mx.array([1.0]), self.alphas_cumprod[:-1]])
        
        # Initialize with default values
        self.num_inference_steps = None
        self.timesteps = None
        self.init_noise_sigma = 1.0
        
    def _get_beta_schedule(self):
        """Get the beta schedule"""
        if self.beta_schedule == "linear":
            return mx.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "scaled_linear":
            # Used by Stable Diffusion
            return mx.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps)**2
        elif self.beta_schedule == "squaredcos_cap_v2":
            return self._betas_for_alpha_bar(lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
    
    def _betas_for_alpha_bar(self, alpha_bar_fn):
        """Helper for squaredcos_cap_v2 schedule"""
        betas = []
        for i in range(self.num_train_timesteps):
            t1 = i / self.num_train_timesteps
            t2 = (i + 1) / self.num_train_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
        return mx.array(betas)
    
    @abstractmethod
    def set_timesteps(self, num_inference_steps: int):
        """Set the discrete timesteps used for the diffusion chain"""
        pass
    
    @abstractmethod
    def step(self, model_output: mx.array, timestep: int, sample: mx.array, **kwargs) -> mx.array:
        """Predict the sample from the previous timestep"""
        pass
    
    def scale_model_input(self, sample: mx.array, timestep: Optional[int] = None) -> mx.array:
        """Scale the denoising model input"""
        return sample


class DDIMScheduler(SchedulerMixin):
    """Denoising Diffusion Implicit Models (DDIM) Scheduler"""
    
    def __init__(self, 
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.00085,
                 beta_end: float = 0.012,
                 beta_schedule: str = "scaled_linear",
                 clip_sample: bool = False,
                 set_alpha_to_one: bool = False,
                 steps_offset: int = 0):
        
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule)
        
        self.clip_sample = clip_sample
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset
        
        # Calculate final alpha cumprod
        final_alpha_cumprod = 1.0 if set_alpha_to_one else self.alphas_cumprod[0]
        self.final_alpha_cumprod = final_alpha_cumprod
    
    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for DDIM sampling"""
        self.num_inference_steps = num_inference_steps
        
        # Create evenly spaced timesteps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (mx.arange(0, num_inference_steps) * step_ratio).round()
        timesteps += self.steps_offset
        
        self.timesteps = timesteps[::-1].copy().astype(mx.int32)
    
    def step(self, 
             model_output: mx.array,
             timestep: int, 
             sample: mx.array,
             eta: float = 0.0,
             use_clipped_model_output: bool = False,
             generator: Optional[mx.random.state] = None,
             variance_noise: Optional[mx.array] = None) -> mx.array:
        """Predict the sample at the previous timestep"""
        
        # Get current and previous alpha_cumprod
        alpha_prod_t = self.alphas_cumprod[timestep]
        
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        if prev_timestep >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        else:
            alpha_prod_t_prev = self.final_alpha_cumprod
            
        beta_prod_t = 1 - alpha_prod_t
        
        # Compute predicted original sample
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        
        if self.clip_sample:
            pred_original_sample = mx.clip(pred_original_sample, -1, 1)
            
        # Compute variance
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance**0.5
        
        if use_clipped_model_output:
            model_output = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5
            
        # Compute predicted previous sample
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2)**0.5 * model_output
        
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError("Cannot pass both generator and variance_noise")
            if variance_noise is None:
                variance_noise = mx.random.normal(sample.shape, dtype=sample.dtype)
                
            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance
            
        return prev_sample
    
    def _get_variance(self, timestep, prev_timestep):
        """Get variance for the timestep"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance


class PNDMScheduler(SchedulerMixin):
    """Pseudo Linear Multistep (PLMS) Scheduler"""
    
    def __init__(self, 
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.00085,
                 beta_end: float = 0.012,
                 beta_schedule: str = "scaled_linear",
                 skip_prk_steps: bool = False,
                 set_alpha_to_one: bool = False,
                 steps_offset: int = 0):
        
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule)
        
        self.skip_prk_steps = skip_prk_steps
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset
        
        # Initialize model outputs list
        self.ets = []
        self.counter = 0
        
        final_alpha_cumprod = 1.0 if set_alpha_to_one else self.alphas_cumprod[0]
        self.final_alpha_cumprod = final_alpha_cumprod
    
    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for PNDM sampling"""
        self.num_inference_steps = num_inference_steps
        
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (mx.arange(0, num_inference_steps) * step_ratio).round()
        timesteps += self.steps_offset
        
        if self.skip_prk_steps:
            # Skip Runge-Kutta steps
            self.prk_timesteps = mx.array([])
            self.plms_timesteps = timesteps[::-1].copy()
        else:
            # Include Runge-Kutta steps for first few iterations
            prk_timesteps = timesteps[-4:].copy()
            prk_timesteps = prk_timesteps[::-1]
            plms_timesteps = timesteps[:-4][::-1].copy()
            
            self.prk_timesteps = prk_timesteps
            self.plms_timesteps = plms_timesteps
            
        timesteps_combined = mx.concatenate([self.prk_timesteps, self.plms_timesteps])
        self.timesteps = timesteps_combined.astype(mx.int32)
        
        # Reset counters
        self.ets = []
        self.counter = 0
    
    def step(self, 
             model_output: mx.array,
             timestep: int,
             sample: mx.array,
             **kwargs) -> mx.array:
        """Predict the sample at the previous timestep"""
        
        if len(self.ets) < 4:
            # Runge-Kutta steps
            self.ets.append(model_output)
            return self._step_prk(model_output, timestep, sample)
        else:
            # PLMS steps
            self.ets.append(model_output)
            if len(self.ets) > 4:
                self.ets.pop(0)
            return self._step_plms(model_output, timestep, sample)
    
    def _step_prk(self, model_output: mx.array, timestep: int, sample: mx.array) -> mx.array:
        """Runge-Kutta step"""
        if self.counter < len(self.prk_timesteps):
            prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        else:
            prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
            
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        
        # Compute predicted original sample
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        
        # Compute predicted previous sample
        pred_sample_direction = (1 - alpha_prod_t_prev)**0.5 * model_output
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        
        self.counter += 1
        return prev_sample
    
    def _step_plms(self, model_output: mx.array, timestep: int, sample: mx.array) -> mx.array:
        """PLMS step"""
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        
        # Compute predicted original sample
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        
        # Compute PLMS coefficients
        plms_coeff = self._get_plms_coefficient()
        
        # Weighted sum of model outputs
        weighted_model_output = sum(coeff * et for coeff, et in zip(plms_coeff, self.ets[-4:]))
        
        # Compute predicted previous sample
        pred_sample_direction = (1 - alpha_prod_t_prev)**0.5 * weighted_model_output
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        
        return prev_sample
    
    def _get_plms_coefficient(self):
        """Get PLMS coefficients"""
        # Adams-Bashforth coefficients for PLMS
        return [55/24, -59/24, 37/24, -9/24]


class LMSScheduler(SchedulerMixin):
    """Linear Multistep Scheduler"""
    
    def __init__(self, 
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.00085,
                 beta_end: float = 0.012,
                 beta_schedule: str = "scaled_linear",
                 order: int = 4):
        
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule)
        
        self.order = order
        self.derivatives = []
        
        # Compute sigmas for LMS
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod)**0.5
        self.log_sigmas = mx.log(self.sigmas)
    
    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for LMS sampling"""
        self.num_inference_steps = num_inference_steps
        
        # Linear interpolation in log space
        sigmas = mx.exp(mx.linspace(
            mx.log(self.sigmas[-1]),
            mx.log(self.sigmas[0]),
            num_inference_steps + 1
        ))
        
        # Append final sigma of 0
        self.sigmas = mx.concatenate([sigmas, mx.array([0.0])])
        self.timesteps = mx.arange(num_inference_steps)
        
        self.derivatives = []
    
    def scale_model_input(self, sample: mx.array, timestep: int) -> mx.array:
        """Scale input by sigma"""
        sigma = self.sigmas[timestep]
        return sample / ((sigma**2 + 1)**0.5)
    
    def step(self, 
             model_output: mx.array,
             timestep: int,
             sample: mx.array,
             order: Optional[int] = None,
             **kwargs) -> mx.array:
        """Predict the sample at the previous timestep using LMS"""
        
        if order is None:
            order = self.order
            
        sigma = self.sigmas[timestep]
        sigma_prev = self.sigmas[timestep + 1]
        
        # Convert model output to derivative
        derivative = (sample - model_output) / sigma
        self.derivatives.append(derivative)
        
        if len(self.derivatives) > order:
            self.derivatives.pop(0)
            
        # Linear multistep method
        prev_sample = sample + (sigma_prev - sigma) * self._linear_multistep_coeff(order, len(self.derivatives) - 1)
        
        return prev_sample
    
    def _linear_multistep_coeff(self, order: int, curr_order: int):
        """Compute linear multistep coefficients"""
        if curr_order == 0:
            return self.derivatives[-1]
        elif curr_order == 1:
            return (3 * self.derivatives[-1] - self.derivatives[-2]) / 2
        elif curr_order == 2:
            return (23 * self.derivatives[-1] - 16 * self.derivatives[-2] + 5 * self.derivatives[-3]) / 12
        elif curr_order == 3:
            return (55 * self.derivatives[-1] - 59 * self.derivatives[-2] + 37 * self.derivatives[-3] - 9 * self.derivatives[-4]) / 24
        else:
            raise NotImplementedError(f"Order {curr_order} not implemented")


def create_scheduler(scheduler_type: str = "ddim", **kwargs) -> SchedulerMixin:
    """Factory function to create appropriate scheduler"""
    
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "ddim":
        return DDIMScheduler(**kwargs)
    elif scheduler_type == "pndm" or scheduler_type == "plms":
        return PNDMScheduler(**kwargs)
    elif scheduler_type == "lms":
        return LMSScheduler(**kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")