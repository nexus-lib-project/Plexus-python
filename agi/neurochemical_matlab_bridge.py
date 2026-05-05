# neurochemical_matlab_bridge.py - MATLAB Integration for Neurochemical Simulation
# Version 1.0: Advanced Mathematical Operations for Brain Simulation
# 
# This module provides a bridge between Python and MATLAB for advanced
# neurochemical simulations. It can work with MATLAB Engine API if available,
# or fall back to pure Python/NumPy implementations.

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import threading
import json
import os

# Try to import MATLAB Engine API
MATLAB_ENGINE_AVAILABLE = False
matlab_engine = "neurochemical_simulation.m"

try:
    import matlab.engine
    MATLAB_ENGINE_AVAILABLE = True
    print("[MATLAB BRIDGE] MATLAB Engine API found. Advanced simulation mode available.")
except ImportError:
    print("[MATLAB BRIDGE] MATLAB Engine API not found. Using NumPy fallback mode.")


# =============================================================================
# MATLAB SIMULATION CONFIGURATION
# =============================================================================

@dataclass
class MatlabSimulationConfig:
    """Configuration for MATLAB-based neurochemical simulations."""
    
    # Simulation parameters
    time_step: float = 0.001  # seconds
    simulation_duration: float = 1.0  # seconds
    
    # Numerical methods
    ode_solver: str = "ode45"  # ode45, ode23, ode15s, etc.
    integration_method: str = "runge_kutta_4"
    
    # Neurochemical model parameters
    diffusion_coefficient: float = 0.1
    reaction_rate: float = 0.05
    
    # Neural dynamics
    membrane_time_constant: float = 10.0  # ms
    synaptic_delay: float = 1.0  # ms
    
    # Enable/disable features
    use_gpu: bool = False
    parallel_processing: bool = True
    verbose: bool = True


# =============================================================================
# NUMPY FALLBACK IMPLEMENTATIONS (When MATLAB is not available)
# =============================================================================

class NumPyNeurochemicalSolver:
    """
    Pure NumPy implementation of neurochemical dynamics.
    Used as fallback when MATLAB is not available.
    """
    
    def __init__(self, config: MatlabSimulationConfig = None):
        self.config = config or MatlabSimulationConfig()
        
    def solve_ode_system(self, initial_conditions: np.ndarray, 
                         derivatives_func: callable,
                         time_span: Tuple[float, float],
                         num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve a system of ODEs using Runge-Kutta 4th order method.
        
        Args:
            initial_conditions: Initial state vector
            derivatives_func: Function that computes derivatives
            time_span: (t_start, t_end) tuple
            num_points: Number of time points
            
        Returns:
            Tuple of (time_points, solution_matrix)
        """
        t_start, t_end = time_span
        dt = (t_end - t_start) / num_points
        
        t = np.linspace(t_start, t_end, num_points)
        y = np.zeros((num_points, len(initial_conditions)))
        y[0] = initial_conditions
        
        for i in range(1, num_points):
            # RK4 method
            k1 = derivatives_func(t[i-1], y[i-1])
            k2 = derivatives_func(t[i-1] + dt/2, y[i-1] + dt*k1/2)
            k3 = derivatives_func(t[i-1] + dt/2, y[i-1] + dt*k2/2)
            k4 = derivatives_func(t[i-1] + dt, y[i-1] + dt*k3)
            
            y[i] = y[i-1] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
        return t, y
    
    def solve_diffusion_equation(self, concentration: np.ndarray,
                                  diffusion_coeff: float,
                                  time_steps: int = 100) -> np.ndarray:
        """
        Solve the diffusion equation for neurochemical spread.
        
        Uses finite difference method for the heat equation:
        ∂C/∂t = D * ∇²C
        
        Args:
            concentration: Initial concentration field (1D, 2D, or 3D array)
            diffusion_coeff: Diffusion coefficient
            time_steps: Number of time steps to simulate
            
        Returns:
            Final concentration field
        """
        C = concentration.copy()
        dt = 0.001
        dx = 1.0
        
        alpha = diffusion_coeff * dt / (dx ** 2)
        
        for _ in range(time_steps):
            if C.ndim == 1:
                # 1D diffusion
                C_new = C.copy()
                C_new[1:-1] = C[1:-1] + alpha * (C[2:] - 2*C[1:-1] + C[:-2])
                C = C_new
            elif C.ndim == 2:
                # 2D diffusion
                C_new = C.copy()
                C_new[1:-1, 1:-1] = C[1:-1, 1:-1] + alpha * (
                    C[2:, 1:-1] + C[:-2, 1:-1] + 
                    C[1:-1, 2:] + C[1:-1, :-2] - 
                    4 * C[1:-1, 1:-1]
                )
                C = C_new
            elif C.ndim == 3:
                # 3D diffusion
                C_new = C.copy()
                C_new[1:-1, 1:-1, 1:-1] = C[1:-1, 1:-1, 1:-1] + alpha * (
                    C[2:, 1:-1, 1:-1] + C[:-2, 1:-1, 1:-1] +
                    C[1:-1, 2:, 1:-1] + C[1:-1, :-2, 1:-1] +
                    C[1:-1, 1:-1, 2:] + C[1:-1, 1:-1, :-2] -
                    6 * C[1:-1, 1:-1, 1:-1]
                )
                C = C_new
                
        return C
    
    def compute_neural_field(self, x: np.ndarray, y: np.ndarray,
                             amplitude: float, sigma: float,
                             center: Tuple[float, float]) -> np.ndarray:
        """
        Compute a neural field using Gaussian basis functions.
        
        Args:
            x, y: Coordinate grids
            amplitude: Field amplitude
            sigma: Spread parameter
            center: (x0, y0) center position
            
        Returns:
            Neural field values
        """
        x0, y0 = center
        return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    def compute_power_spectrum(self, signal: np.ndarray, 
                                sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum of a neural signal using FFT.
        
        Args:
            signal: Time series signal
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (frequencies, power)
        """
        n = len(signal)
        fft_result = np.fft.fft(signal)
        power = np.abs(fft_result[:n//2])**2
        frequencies = np.fft.fftfreq(n, 1/sampling_rate)[:n//2]
        return frequencies, power
    
    def compute_coherence(self, signal1: np.ndarray, signal2: np.ndarray,
                          sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coherence between two signals.
        
        Args:
            signal1, signal2: Time series signals
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (frequencies, coherence)
        """
        n = len(signal1)
        
        # Compute FFTs
        fft1 = np.fft.fft(signal1)
        fft2 = np.fft.fft(signal2)
        
        # Cross-spectral density
        csd = fft1 * np.conj(fft2)
        
        # Power spectral densities
        psd1 = np.abs(fft1)**2
        psd2 = np.abs(fft2)**2
        
        # Coherence
        coherence = np.abs(csd)**2 / (psd1 * psd2 + 1e-10)
        frequencies = np.fft.fftfreq(n, 1/sampling_rate)[:n//2]
        
        return frequencies[:n//2], coherence[:n//2]
    
    def hodgkin_huxley_neuron(self, t: np.ndarray, I_ext: float = 10.0) -> Dict[str, np.ndarray]:
        """
        Simulate a Hodgkin-Huxley neuron model.
        
        Args:
            t: Time array
            I_ext: External current
            
        Returns:
            Dictionary with V, m, h, n state variables
        """
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        
        # Initialize state variables
        V = np.zeros_like(t)
        m = np.zeros_like(t)
        h = np.zeros_like(t)
        n = np.zeros_like(t)
        
        # Initial conditions
        V[0] = -65.0  # mV
        m[0] = 0.05
        h[0] = 0.6
        n[0] = 0.32
        
        # HH parameters
        g_Na = 120.0  # mS/cm²
        g_K = 36.0
        g_L = 0.3
        E_Na = 50.0  # mV
        E_K = -77.0
        E_L = -54.4
        
        C_m = 1.0  # µF/cm²
        
        for i in range(1, len(t)):
            # Rate constants
            alpha_m = 0.1 * (V[i-1] + 40) / (1 - np.exp(-(V[i-1] + 40) / 10))
            beta_m = 4.0 * np.exp(-(V[i-1] + 65) / 18)
            
            alpha_h = 0.07 * np.exp(-(V[i-1] + 65) / 20)
            beta_h = 1.0 / (1 + np.exp(-(V[i-1] + 35) / 10))
            
            alpha_n = 0.01 * (V[i-1] + 55) / (1 - np.exp(-(V[i-1] + 55) / 10))
            beta_n = 0.125 * np.exp(-(V[i-1] + 65) / 80)
            
            # Update gating variables
            m[i] = m[i-1] + dt * (alpha_m * (1 - m[i-1]) - beta_m * m[i-1])
            h[i] = h[i-1] + dt * (alpha_h * (1 - h[i-1]) - beta_h * h[i-1])
            n[i] = n[i-1] + dt * (alpha_n * (1 - n[i-1]) - beta_n * n[i-1])
            
            # Ionic currents
            I_Na = g_Na * m[i]**3 * h[i] * (V[i-1] - E_Na)
            I_K = g_K * n[i]**4 * (V[i-1] - E_K)
            I_L = g_L * (V[i-1] - E_L)
            
            # Membrane potential
            V[i] = V[i-1] + dt * (I_ext - I_Na - I_K - I_L) / C_m
            
        return {'V': V, 'm': m, 'h': h, 'n': n}
    
    def izhikevich_neuron(self, t: np.ndarray, I_ext: np.ndarray,
                          a: float = 0.02, b: float = 0.2,
                          c: float = -65.0, d: float = 8.0) -> Dict[str, np.ndarray]:
        """
        Simulate an Izhikevich neuron model.
        
        Args:
            t: Time array
            I_ext: External current array
            a, b, c, d: Model parameters
            
        Returns:
            Dictionary with V (membrane potential) and u (recovery variable)
        """
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        
        V = np.zeros_like(t)
        u = np.zeros_like(t)
        
        V[0] = c
        u[0] = b * c
        
        for i in range(1, len(t)):
            # Check for spike
            if V[i-1] >= 30:
                V[i-1] = 30  # Cap at spike threshold
                V[i] = c
                u[i] = u[i-1] + d
            else:
                # Izhikevich equations
                dV = 0.04 * V[i-1]**2 + 5 * V[i-1] + 140 - u[i-1] + I_ext[i-1]
                du = a * (b * V[i-1] - u[i-1])
                
                V[i] = V[i-1] + dt * dV
                u[i] = u[i-1] + dt * du
                
        return {'V': V, 'u': u}


# =============================================================================
# MATLAB ENGINE WRAPPER (When MATLAB is available)
# =============================================================================

class MatlabNeurochemicalSolver:
    """
    MATLAB Engine-based implementation of neurochemical dynamics.
    Provides advanced numerical methods and visualization.
    """
    
    def __init__(self, config: MatlabSimulationConfig = None):
        self.config = config or MatlabSimulationConfig()
        self.engine = None
        self._initialize_matlab()
        
    def _initialize_matlab(self):
        """Initialize MATLAB engine connection."""
        global matlab_engine, MATLAB_ENGINE_AVAILABLE
        
        if MATLAB_ENGINE_AVAILABLE:
            try:
                if matlab_engine is None:
                    matlab_engine = matlab.engine.start_matlab()
                self.engine = matlab_engine
                print("[MATLAB BRIDGE] MATLAB engine connected successfully.")
            except Exception as e:
                print(f"[MATLAB BRIDGE ERROR] Failed to start MATLAB engine: {e}")
                self.engine = None
        else:
            self.engine = None
            
    def solve_ode_system(self, initial_conditions: np.ndarray,
                         derivatives_func: callable,
                         time_span: Tuple[float, float],
                         num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE system using MATLAB's ode45 or other solvers.
        Falls back to NumPy if MATLAB is not available.
        """
        if self.engine is None:
            # Fall back to NumPy
            numpy_solver = NumPyNeurochemicalSolver(self.config)
            return numpy_solver.solve_ode_system(initial_conditions, derivatives_func, 
                                                  time_span, num_points)
        
        try:
            # Convert to MATLAB format
            y0 = matlab.double(initial_conditions.tolist())
            tspan = matlab.double(list(time_span))
            
            # Define the ODE function as MATLAB string
            # This is a simplified version - in practice, you'd need to
            # pass the derivatives function properly
            ode_func = "@(t,y) y * 0.1;"  # Placeholder
            
            # Call MATLAB ode45
            result = self.engine.eval(f"[T,Y] = ode45({ode_func}, {tspan}, {y0});", nargout=2)
            
            # Convert back to NumPy
            T = np.array(result[0])
            Y = np.array(result[1])
            
            return T, Y
            
        except Exception as e:
            print(f"[MATLAB BRIDGE ERROR] ODE solve failed: {e}")
            # Fall back to NumPy
            numpy_solver = NumPyNeurochemicalSolver(self.config)
            return numpy_solver.solve_ode_system(initial_conditions, derivatives_func,
                                                  time_span, num_points)


# =============================================================================
# NEUROCHEMICAL SIMULATION ENGINE
# =============================================================================

class NeurochemicalMatlabSimulation:
    """
    Main simulation engine that combines MATLAB/NumPy solvers
    for comprehensive neurochemical simulations.
    """
    
    def __init__(self, config: MatlabSimulationConfig = None):
        self.config = config or MatlabSimulationConfig()
        
        # Initialize appropriate solver
        if MATLAB_ENGINE_AVAILABLE:
            self.solver = MatlabNeurochemicalSolver(self.config)
        else:
            self.solver = NumPyNeurochemicalSolver(self.config)
            
        # Simulation state
        self.current_time = 0.0
        self.neurochemical_state = {}
        self.neural_state = {}
        
        # History tracking
        self.state_history = []
        self.max_history = 1000
        
    def simulate_neurochemical_dynamics(self, 
                                        initial_concentrations: Dict[str, float],
                                        time_duration: float = 1.0,
                                        input_signals: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Simulate the dynamics of multiple neurochemicals over time.
        
        Args:
            initial_concentrations: Dict of neurochemical name -> initial concentration
            time_duration: Duration of simulation in seconds
            input_signals: Optional dict of neurochemical name -> input signal array
            
        Returns:
            Dict of neurochemical name -> concentration time series
        """
        num_steps = int(time_duration / self.config.time_step)
        t = np.linspace(0, time_duration, num_steps)
        
        results = {}
        
        for name, initial_conc in initial_concentrations.items():
            # Define ODE for this neurochemical
            def derivatives(t, y, name=name):
                # Simplified dynamics: synthesis - degradation + input
                baseline = 0.5
                synthesis_rate = 0.1
                degradation_rate = 0.05
                
                # Get input signal if available
                input_val = 0.0
                if input_signals and name in input_signals:
                    idx = min(int(t / self.config.time_step), len(input_signals[name]) - 1)
                    input_val = input_signals[name][idx]
                    
                dydt = synthesis_rate * (baseline - y[0]) - degradation_rate * y[0] + input_val
                return np.array([dydt])
            
            # Solve ODE
            y0 = np.array([initial_conc])
            _, y = self.solver.solve_ode_system(y0, derivatives, (0, time_duration), num_steps)
            results[name] = y[:, 0]
            
        return results
    
    def simulate_neural_population(self, 
                                   num_neurons: int = 100,
                                   time_duration: float = 1.0,
                                   connectivity: np.ndarray = None,
                                   external_input: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Simulate a population of interconnected neurons.
        
        Args:
            num_neurons: Number of neurons in the population
            time_duration: Simulation duration in seconds
            connectivity: Connectivity matrix (num_neurons x num_neurons)
            external_input: External input to each neuron over time
            
        Returns:
            Dict with spike times, membrane potentials, etc.
        """
        dt = self.config.time_step
        num_steps = int(time_duration / dt)
        t = np.linspace(0, time_duration, num_steps)
        
        # Initialize neurons
        V = np.random.randn(num_neurons) * 5 - 65  # Random initial potentials
        spikes = [[] for _ in range(num_neurons)]
        
        # Default connectivity (random sparse)
        if connectivity is None:
            connectivity = np.random.randn(num_neurons, num_neurons) * 0.1
            connectivity *= (np.random.rand(num_neurons, num_neurons) < 0.1)
            
        # Default external input
        if external_input is None:
            external_input = np.random.randn(num_steps, num_neurons) * 2
            
        # Simulate
        V_history = np.zeros((num_steps, num_neurons))
        
        for i in range(num_steps):
            # Synaptic input
            synaptic_input = np.dot(connectivity, (V > -50).astype(float))
            
            # Update membrane potential (leaky integrate-and-fire)
            dV = (-V - (-65) + external_input[i] + synaptic_input * 10) / 20
            V = V + dt * dV
            
            # Check for spikes
            spiked = V > -50
            for n in range(num_neurons):
                if spiked[n]:
                    spikes[n].append(t[i])
                    V[n] = -65  # Reset
                    
            V_history[i] = V
            
        return {
            'time': t,
            'membrane_potentials': V_history,
            'spike_times': spikes,
            'firing_rates': np.array([len(s) / time_duration for s in spikes])
        }
    
    def compute_brain_waves(self, signal: np.ndarray, 
                            sampling_rate: float = 1000.0) -> Dict[str, float]:
        """
        Compute brain wave power in different frequency bands.
        
        Args:
            signal: Neural signal (e.g., LFP or EEG)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dict with power in each frequency band
        """
        frequencies, power = self.solver.compute_power_spectrum(signal, sampling_rate)
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'high_gamma': (100, 200)
        }
        
        band_power = {}
        for band_name, (low, high) in bands.items():
            mask = (frequencies >= low) & (frequencies < high)
            band_power[band_name] = np.mean(power[mask]) if np.any(mask) else 0.0
            
        return band_power
    
    def simulate_receptor_binding(self, 
                                  ligand_concentration: float,
                                  receptor_density: float = 1.0,
                                  affinity: float = 0.5,
                                  time_duration: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Simulate receptor-ligand binding dynamics.
        
        Uses the law of mass action:
        d[LR]/dt = k_on * [L] * [R] - k_off * [LR]
        
        Args:
            ligand_concentration: Concentration of ligand
            receptor_density: Total receptor density
            affinity: Binding affinity (K_d)
            time_duration: Simulation duration
            
        Returns:
            Dict with bound and free receptor concentrations over time
        """
        num_steps = int(time_duration / self.config.time_step)
        t = np.linspace(0, time_duration, num_steps)
        
        # Rate constants
        k_on = affinity * 10
        k_off = affinity
        
        # Initial conditions
        LR = np.zeros(num_steps)  # Bound receptors
        R = np.ones(num_steps) * receptor_density  # Free receptors
        
        for i in range(1, num_steps):
            dt = t[i] - t[i-1]
            
            # Binding kinetics
            dLR = k_on * ligand_concentration * R[i-1] - k_off * LR[i-1]
            
            LR[i] = LR[i-1] + dt * dLR
            R[i] = receptor_density - LR[i]
            
        return {
            'time': t,
            'bound_receptors': LR,
            'free_receptors': R,
            'occupancy': LR / receptor_density
        }
    
    def simulate_synaptic_transmission(self,
                                       presynaptic_spikes: List[float],
                                       time_duration: float = 1.0,
                                       release_probability: float = 0.5,
                                       neurotransmitter_amount: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Simulate synaptic transmission events.
        
        Args:
            presynaptic_spikes: List of presynaptic spike times
            time_duration: Simulation duration
            release_probability: Probability of vesicle release per spike
            neurotransmitter_amount: Amount of neurotransmitter per release
            
        Returns:
            Dict with neurotransmitter concentration and postsynaptic response
        """
        num_steps = int(time_duration / self.config.time_step)
        t = np.linspace(0, time_duration, num_steps)
        dt = self.config.time_step
        
        # Neurotransmitter concentration
        nt_concentration = np.zeros(num_steps)
        
        # Postsynaptic response (simplified)
        ps_response = np.zeros(num_steps)
        
        # Process each spike
        for spike_time in presynaptic_spikes:
            if spike_time > time_duration:
                continue
                
            # Determine if release occurs
            if np.random.random() < release_probability:
                spike_idx = int(spike_time / dt)
                
                # Add neurotransmitter (with decay)
                for i in range(spike_idx, num_steps):
                    decay = np.exp(-(t[i] - spike_time) / 0.01)  # 10ms decay
                    nt_concentration[i] += neurotransmitter_amount * decay
                    
                # Postsynaptic response
                for i in range(spike_idx, num_steps):
                    # AMPA-like fast response
                    ampa = np.exp(-(t[i] - spike_time) / 0.002) * (1 - np.exp(-(t[i] - spike_time) / 0.0005))
                    # NMDA-like slow response
                    nmda = np.exp(-(t[i] - spike_time) / 0.1) * (1 - np.exp(-(t[i] - spike_time) / 0.01))
                    
                    ps_response[i] += ampa * 0.7 + nmda * 0.3
                    
        return {
            'time': t,
            'neurotransmitter': nt_concentration,
            'postsynaptic_response': ps_response
        }
    
    def compute_neurochemical_gradient(self,
                                       source_concentration: float,
                                       distance: np.ndarray,
                                       diffusion_coeff: float = 0.1) -> np.ndarray:
        """
        Compute the concentration gradient of a neurochemical.
        
        Args:
            source_concentration: Concentration at source
            distance: Distance array from source
            diffusion_coeff: Diffusion coefficient
            
        Returns:
            Concentration at each distance
        """
        # Steady-state solution to diffusion equation
        # C(r) = C_0 * exp(-r / lambda) where lambda is diffusion length
        diffusion_length = np.sqrt(diffusion_coeff * 1.0)  # Assuming unit time
        
        concentration = source_concentration * np.exp(-distance / diffusion_length)
        
        return concentration
    
    def get_simulation_report(self) -> Dict[str, Any]:
        """Generate a report of the current simulation state."""
        return {
            'current_time': self.current_time,
            'config': {
                'time_step': self.config.time_step,
                'ode_solver': self.config.ode_solver,
                'matlab_available': MATLAB_ENGINE_AVAILABLE
            },
            'neurochemical_state': self.neurochemical_state.copy(),
            'neural_state': self.neural_state.copy(),
            'history_length': len(self.state_history)
        }


# =============================================================================
# GLOBAL INSTANCE AND HELPER FUNCTIONS
# =============================================================================

# Global simulation instance
_simulation_instance = None

def get_simulation_instance(config: MatlabSimulationConfig = None) -> NeurochemicalMatlabSimulation:
    """Get or create the global simulation instance."""
    global _simulation_instance
    if _simulation_instance is None:
        _simulation_instance = NeurochemicalMatlabSimulation(config)
    return _simulation_instance

def run_neurochemical_simulation(concentrations: Dict[str, float],
                                 duration: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Convenience function to run a neurochemical simulation.
    
    Args:
        concentrations: Initial concentrations
        duration: Simulation duration
        
    Returns:
        Simulation results
    """
    sim = get_simulation_instance()
    return sim.simulate_neurochemical_dynamics(concentrations, duration)

def compute_neural_activity(num_neurons: int = 100,
                            duration: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Convenience function to simulate neural activity.
    
    Args:
        num_neurons: Number of neurons
        duration: Simulation duration
        
    Returns:
        Simulation results
    """
    sim = get_simulation_instance()
    return sim.simulate_neural_population(num_neurons, duration)

def analyze_brain_waves(signal: np.ndarray, 
                        sampling_rate: float = 1000.0) -> Dict[str, float]:
    """
    Convenience function to analyze brain waves.
    
    Args:
        signal: Neural signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Power in each frequency band
    """
    sim = get_simulation_instance()
    return sim.compute_brain_waves(signal, sampling_rate)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def initialize_matlab_bridge():
    """Initialize the MATLAB bridge module."""
    global MATLAB_ENGINE_AVAILABLE, matlab_engine
    
    print("=" * 60)
    print("[MATLAB BRIDGE] Neurochemical Simulation Module v1.0")
    print("=" * 60)
    
    if MATLAB_ENGINE_AVAILABLE:
        print("[MATLAB BRIDGE] MATLAB Engine API detected.")
        print("[MATLAB BRIDGE] Advanced simulation features enabled.")
    else:
        print("[MATLAB BRIDGE] Using NumPy fallback mode.")
        print("[MATLAB BRIDGE] Core simulation features available.")
        
    print("[MATLAB BRIDGE] Module initialized successfully.")
    print("=" * 60)


# Run initialization
initialize_matlab_bridge()
