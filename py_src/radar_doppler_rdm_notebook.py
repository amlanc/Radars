"""
RADAR FUNDAMENTALS & SIMULATOR - Part 2
========================================
Doppler Processing & Range-Doppler Maps

Target: Build complete RDM generation pipeline
Timeline: Weekend 1 (Oct 5-6) continuation

Prerequisites: Complete Part 1 (Range-FFT understanding)
"""

# %%
# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fft2, fftshift, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Visualization setup
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# %% [markdown]
# ---
# # Part 2: Doppler Processing & Range-Doppler Maps
# 
# ## Recap from Part 1
# 
# You now understand:
# - ✅ Range-FFT: Time samples → Range bins (single pulse)
# - ✅ I/Q signals preserve phase information
# - ✅ Radar parameters: PRF, bandwidth, resolution
# 
# ## What's Next
# 
# A **single pulse** gives you range. But to get **velocity**, you need:
# - Multiple coherent pulses (same phase reference)
# - Doppler-FFT across pulses
# - The result: **Range-Doppler Map (RDM)** - a 2D image showing everything
# 
# ---
# # 1. Why We Need Doppler: The Moving Target Problem
# 
# ## The Challenge
# 
# Imagine a stationary clutter (building) at 1000m and a moving drone also at 1000m.
# - **Range-FFT alone:** Both show up in the same range bin - indistinguishable!
# - **Solution:** Use Doppler frequency to separate them
# 
# ## Doppler Shift Physics
# 
# **Moving target causes frequency shift:**
# ```
# f_doppler = 2 * v_radial / λ
# 
# Where:
# - v_radial = velocity toward radar (positive) or away (negative) [m/s]
# - λ = wavelength [m]
# 
# Example: v = 50 m/s, λ = 0.03 m (10 GHz)
# f_doppler = 2 * 50 / 0.03 = 3,333 Hz
# ```
# 
# **Why the factor of 2?**
# - Signal travels TO target (Doppler shift × 1)
# - Reflects back FROM target (Doppler shift × 1 again)
# - Total: 2× the shift
# 
# ## How Doppler Appears in Pulse-Doppler Radar
# 
# **Key insight:**
# - Pulse 1: Target at range R → echo has phase φ₁
# - Pulse 2: Target moved → echo has phase φ₂ = φ₁ + Δφ
# - Pulse 3: Target moved more → echo has phase φ₃ = φ₁ + 2Δφ
# - ...
# 
# **Phase change between pulses:**
# ```
# Δφ = 2π * f_doppler * PRI
#    = 4π * v_radial * PRI / λ
# ```
# 
# **This phase progression IS the Doppler signature!**
# - Stationary target: φ stays constant across pulses
# - Moving target: φ changes linearly across pulses
# - FFT across pulses → frequency → velocity

# %%
def visualize_doppler_concept():
    """
    Demonstrate how Doppler appears as phase change across pulses
    """
    # Parameters
    num_pulses = 64
    PRI = 100e-6  # 100 μs (10 kHz PRF)
    wavelength = 0.03  # 3 cm (10 GHz)
    
    # Three targets at same range, different velocities
    velocities = [0, 25, -25]  # m/s (stationary, approaching, receding)
    colors = ['blue', 'red', 'green']
    labels = ['Stationary (0 m/s)', 'Approaching (+25 m/s)', 'Receding (-25 m/s)']
    
    pulse_times = np.arange(num_pulses) * PRI
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for v, color, label in zip(velocities, colors, labels):
        # Doppler frequency
        f_d = 2 * v / wavelength
        
        # Phase progression across pulses
        phase = 2 * np.pi * f_d * pulse_times
        
        # Complex signal at each pulse (same range bin)
        signal = np.exp(1j * phase)
        
        # Plot phase progression
        axes[0, 0].plot(pulse_times * 1e6, np.angle(signal), 
                       marker='o', label=label, color=color, linewidth=2)
        
        # Plot I/Q constellation
        axes[0, 1].plot(signal.real, signal.imag, 
                       marker='o', label=label, color=color, alpha=0.6)
        
        # FFT to get Doppler spectrum
        spectrum = fftshift(fft(signal))
        freqs = fftshift(fftfreq(num_pulses, PRI))
        
        axes[1, 0].plot(freqs, np.abs(spectrum), 
                       label=label, color=color, linewidth=2)
    
    # Formatting
    axes[0, 0].set_xlabel('Time (μs)')
    axes[0, 0].set_ylabel('Phase (radians)')
    axes[0, 0].set_title('Phase Progression Across Pulses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('I (In-phase)')
    axes[0, 1].set_ylabel('Q (Quadrature)')
    axes[0, 1].set_title('I-Q Constellation (Same Range, Different Velocities)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    axes[1, 0].set_xlabel('Doppler Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Doppler Spectrum (FFT Across Pulses)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([-8000, 8000])
    
    # Convert Doppler freq to velocity for second x-axis
    ax2 = axes[1, 0].twiny()
    ax2.set_xlim(np.array(axes[1, 0].get_xlim()) * wavelength / 2)
    ax2.set_xlabel('Velocity (m/s)', color='gray')
    
    # Summary text
    summary = f"""Doppler Frequencies:
    Stationary: {2*velocities[0]/wavelength:.0f} Hz
    Approaching: {2*velocities[1]/wavelength:.0f} Hz  
    Receding: {2*velocities[2]/wavelength:.0f} Hz"""
    
    axes[1, 1].text(0.1, 0.5, summary, fontsize=12, family='monospace',
                   verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/amlanchatterjee/doppler_concept.png', dpi=150, bbox_inches='tight')
    plt.show()

# Uncomment to run:
# visualize_doppler_concept()

# %% [markdown]
# ---
# # 2. The Range-Doppler Map (RDM): Bringing It All Together
# 
# ## Conceptual Framework
# 
# **Data structure for pulse-Doppler radar:**
# ```
# Raw data matrix: [num_pulses × num_samples]
# 
# Each row = one pulse (fast-time samples)
# Each column = same sample index across pulses (slow-time)
# ```
# 
# **Two-stage FFT:**
# ```
#                    Range-FFT (across columns)
# Raw I/Q data  ──────────────────────────────→  Range bins
#       ↓                                            ↓
#       │                                            │
#       │         Doppler-FFT (across rows)         │
#       └────────────────────────────────────────────┘
#                           ↓
#                   Range-Doppler Map
#                 [Doppler bins × Range bins]
# ```
# 
# **Or equivalently, 2D FFT:**
# ```
# RDM = FFT2D(raw_data_matrix)
# ```
# 
# ## Mathematical Formulation
# 
# **Signal from single target:**
# ```
# s[m, n] = A * exp(j * 2π * (f_r * n/f_s + f_d * m * PRI))
# 
# Where:
# - m = pulse index (0 to M-1)
# - n = sample index within pulse (0 to N-1)
# - f_r = range frequency (from time delay)
# - f_d = Doppler frequency (from velocity)
# ```
# 
# **After 2D FFT:**
# ```
# RDM[k_d, k_r] = |FFT2D(s[m,n])|
# 
# Peak at:
# - k_r ∝ range
# - k_d ∝ velocity
# ```
# 
# ## Range and Velocity Mapping
# 
# **Range axis:**
# ```
# Range[k] = k * c / (2 * f_s)
# 
# where k = 0, 1, ..., N-1 (range bins)
# ```
# 
# **Velocity axis:**
# ```
# Velocity[m] = (m - M/2) * λ / (2 * M * PRI)
#             = (m - M/2) * PRF * λ / (2 * M)
# 
# where m = 0, 1, ..., M-1 (Doppler bins, shifted to center zero)
# ```

# %%
class PulseDopplerRadar:
    """
    Complete pulse-Doppler radar simulator with RDM generation
    """
    
    def __init__(self, f_c=10e9, PRF=10e3, bandwidth=150e6, 
                 num_pulses=128, samples_per_pulse=512):
        """
        Parameters:
        - f_c: Carrier frequency (Hz)
        - PRF: Pulse Repetition Frequency (Hz)
        - bandwidth: Signal bandwidth (Hz)
        - num_pulses: Number of coherent pulses (CPI length)
        - samples_per_pulse: Samples in each pulse
        """
        self.f_c = f_c
        self.PRF = PRF
        self.B = bandwidth
        self.M = num_pulses  # Slow-time dimension
        self.N = samples_per_pulse  # Fast-time dimension
        
        # Derived parameters
        self.c = 3e8
        self.wavelength = self.c / self.f_c
        self.PRI = 1 / self.PRF
        self.f_s = 2 * self.B  # Sampling rate
        self.tau = 1e-6  # Pulse width (1 μs)
        
        # Axes
        self.range_axis = np.arange(self.N) * self.c / (2 * self.f_s)
        self.time_fast = np.arange(self.N) / self.f_s
        self.time_slow = np.arange(self.M) * self.PRI
        
        # Doppler axis (centered at zero velocity)
        doppler_bins = np.arange(self.M) - self.M // 2
        self.velocity_axis = doppler_bins * self.wavelength * self.PRF / (2 * self.M)
        
        # Initialize data matrix
        self.data_matrix = np.zeros((self.M, self.N), dtype=complex)
        
    def add_target(self, range_m, velocity_ms, rcs=1.0):
        """
        Add a point target to the simulation
        
        Parameters:
        - range_m: Target range (meters)
        - velocity_ms: Target radial velocity (m/s, positive = approaching)
        - rcs: Radar cross section (relative amplitude)
        """
        # Range bin and Doppler frequency
        range_bin = int(range_m * 2 * self.f_s / self.c)
        f_doppler = 2 * velocity_ms / self.wavelength
        
        # Generate signal across all pulses
        for m in range(self.M):
            # Range component (phase from range)
            phase_range = -4 * np.pi * range_m / self.wavelength
            
            # Doppler component (phase progression across pulses)
            phase_doppler = 2 * np.pi * f_doppler * m * self.PRI
            
            # Total phase
            phase_total = phase_range + phase_doppler
            
            # Add pulse at correct range bin
            pulse_samples = int(self.tau * self.f_s)
            if range_bin + pulse_samples < self.N:
                self.data_matrix[m, range_bin:range_bin + pulse_samples] += \
                    rcs * np.exp(1j * phase_total)
    
    def add_noise(self, snr_db=20):
        """Add white Gaussian noise"""
        signal_power = np.mean(np.abs(self.data_matrix)**2)
        noise_power = signal_power / (10**(snr_db/10))
        
        noise = np.sqrt(noise_power/2) * \
                (np.random.randn(self.M, self.N) + 1j * np.random.randn(self.M, self.N))
        
        self.data_matrix += noise
    
    def generate_rdm(self, window='hamming'):
        """
        Generate Range-Doppler Map using 2D FFT
        
        Returns:
        - rdm: Range-Doppler Map (Doppler bins × Range bins)
        - range_axis: Range values (meters)
        - velocity_axis: Velocity values (m/s)
        """
        # Apply 2D windowing to reduce sidelobes
        if window == 'hamming':
            window_range = np.hamming(self.N)
            window_doppler = np.hamming(self.M)
            window_2d = np.outer(window_doppler, window_range)
        else:
            window_2d = 1.0
        
        data_windowed = self.data_matrix * window_2d
        
        # 2D FFT: Range-FFT then Doppler-FFT
        rdm_raw = fft2(data_windowed)
        
        # Shift Doppler axis to center zero velocity
        rdm = fftshift(rdm_raw, axes=0)
        
        # Magnitude (power)
        rdm_magnitude = np.abs(rdm)
        
        return rdm_magnitude, self.range_axis, self.velocity_axis
    
    def plot_rdm(self, rdm, range_axis, velocity_axis, 
                 range_max_km=10, dynamic_range_db=60):
        """
        Visualize the Range-Doppler Map
        """
        # Convert to dB scale
        rdm_db = 20 * np.log10(rdm + 1e-10)
        rdm_db -= np.max(rdm_db)  # Normalize to peak
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use only valid range bins
        range_bins = range_axis < range_max_km * 1000
        
        im = ax.imshow(
            rdm_db[:, range_bins],
            aspect='auto',
            extent=[0, range_max_km, 
                   velocity_axis[0], velocity_axis[-1]],
            cmap='jet',
            vmin=-dynamic_range_db,
            vmax=0,
            origin='lower'
        )
        
        ax.set_xlabel('Range (km)', fontsize=12)
        ax.set_ylabel('Velocity (m/s)', fontsize=12)
        ax.set_title('Range-Doppler Map (RDM)', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)', fontsize=12)
        
        # Reference lines
        ax.axhline(0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3, color='white', linestyle=':')
        
        plt.tight_layout()
        return fig

# %% [markdown]
# ---
# # 3. Example: Generate Your First RDM
# 
# Let's create a scenario with multiple targets:
# - Stationary clutter at various ranges
# - Moving drone
# - Bird
# - Aircraft

# %%
def scenario_multi_target():
    """
    Realistic counter-UAS scenario
    """
    # Initialize radar
    radar = PulseDopplerRadar(
        f_c=10e9,          # 10 GHz X-band
        PRF=10e3,          # 10 kHz
        bandwidth=150e6,   # 150 MHz
        num_pulses=128,    # 128 pulses in CPI
        samples_per_pulse=512
    )
    
    # Add targets
    print("Creating scenario...")
    print("\nTargets:")
    
    # Stationary clutter (buildings, ground)
    for r in [500, 1000, 2000, 3000]:
        radar.add_target(range_m=r, velocity_ms=0, rcs=0.8)
        print(f"  Clutter at {r}m, velocity: 0 m/s")
    
    # Moving targets
    # 1. Drone (approaching)
    radar.add_target(range_m=1500, velocity_ms=15, rcs=0.01)
    print(f"  Drone at 1500m, velocity: +15 m/s (approaching)")
    
    # 2. Bird (slower, different range)
    radar.add_target(range_m=800, velocity_ms=8, rcs=0.005)
    print(f"  Bird at 800m, velocity: +8 m/s (approaching)")
    
    # 3. Small aircraft (faster, higher RCS)
    radar.add_target(range_m=5000, velocity_ms=-50, rcs=1.0)
    print(f"  Aircraft at 5000m, velocity: -50 m/s (receding)")
    
    # Add noise
    radar.add_noise(snr_db=15)
    
    # Generate RDM
    print("\nGenerating Range-Doppler Map...")
    rdm, range_axis, velocity_axis = radar.generate_rdm()
    
    # DEBUG: Check RDM statistics
    print(f"\nDEBUG RDM:")
    print(f"  RDM shape: {rdm.shape}")
    print(f"  RDM max: {np.max(rdm):.2e}")
    print(f"  RDM mean: {np.mean(rdm):.2e}")
    print(f"  Max position: {np.unravel_index(np.argmax(rdm), rdm.shape)}")
    
    # Find peaks in RDM
    rdm_flat = rdm.flatten()
    top_indices = np.argsort(rdm_flat)[-10:]  # Top 10 values
    print(f"  Top 10 peak positions (doppler_bin, range_bin):")
    for idx in top_indices[::-1]:
        pos = np.unravel_index(idx, rdm.shape)
        print(f"    {pos}: velocity={velocity_axis[pos[0]]:.1f} m/s, range={range_axis[pos[1]]:.1f} m, magnitude={rdm[pos]:.2e}")
    
    # Visualize
    fig = radar.plot_rdm(rdm, range_axis, velocity_axis, range_max_km=8)
    plt.savefig('/Users/amlanchatterjee/rdm_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print system parameters
    print("\nRadar Parameters:")
    print(f"  Range resolution: {radar.c/(2*radar.B):.2f} m")
    print(f"  Velocity resolution: {radar.wavelength*radar.PRF/(2*radar.M):.2f} m/s")
    print(f"  Max unambiguous range: {radar.c/(2*radar.PRF)/1000:.2f} km")
    print(f"  Max unambiguous velocity: {radar.wavelength*radar.PRF/4:.2f} m/s")
    
    return radar, rdm, range_axis, velocity_axis

# Uncomment to run:
# radar, rdm, range_axis, velocity_axis = scenario_multi_target()

# %% [markdown]
# ---
# # 4. Understanding the RDM: What You're Looking At
# 
# When you look at the RDM image:
# 
# ## Axes
# - **X-axis (horizontal):** Range (distance from radar)
# - **Y-axis (vertical):** Velocity (positive = approaching, negative = receding)
# 
# ## What You See
# 
# **Bright spots = targets**
# - Position tells you range AND velocity simultaneously
# - Brightness ∝ target strength (RCS)
# 
# **Vertical line at velocity = 0:**
# - Stationary clutter (buildings, ground, trees)
# - This is why we need MTI (Moving Target Indication) to filter it out
# 
# **Spots away from zero velocity:**
# - Moving targets (drones, birds, aircraft, vehicles)
# - Horizontal position = range
# - Vertical position = velocity
# 
# ## Resolution Limits
# 
# **Range resolution (can you separate two targets in range?):**
# ```
# ΔR = c / (2 * B)
# 
# Example: B = 150 MHz → ΔR = 1 m
# ```
# 
# **Velocity resolution (can you separate two targets in velocity?):**
# ```
# Δv = λ * PRF / (2 * M)
# 
# Example: λ = 3 cm, PRF = 10 kHz, M = 128
# Δv = 0.03 * 10000 / (2 * 128) = 1.17 m/s
# ```

# %% [markdown]
# ---
# # 5. MTI: Moving Target Indication
# 
# ## The Problem
# 
# Clutter (stationary returns) can be 40-60 dB stronger than moving targets!
# - Ground returns
# - Buildings
# - Trees, vegetation
# 
# **They overwhelm the moving targets you care about.**
# 
# ## The Solution: Clutter Filtering
# 
# **Simple approach: High-pass filter in Doppler dimension**
# - Remove DC component (zero velocity)
# - Keep only non-zero Doppler frequencies
# 
# ```python
# # After Range-FFT, before Doppler-FFT:
# for each range bin:
#     Remove mean across pulses (DC removal)
#     Apply high-pass filter
# ```

# %%
def apply_mti(radar_obj, rdm):
    """
    Apply Moving Target Indication (MTI) filter
    
    Simple 2-pulse canceller approach
    """
    # Clone data matrix
    data_mti = radar_obj.data_matrix.copy()
    
    # Simple MTI: Subtract adjacent pulses
    # This cancels stationary returns, preserves moving targets
    data_mti[1:, :] = data_mti[1:, :] - data_mti[:-1, :]
    
    # Generate RDM with MTI
    window_range = np.hamming(radar_obj.N)
    window_doppler = np.hamming(radar_obj.M)
    window_2d = np.outer(window_doppler, window_range)
    
    rdm_mti = fft2(data_mti * window_2d)
    rdm_mti = fftshift(rdm_mti, axes=0)
    
    return np.abs(rdm_mti)

# Example with MTI
def compare_with_without_mti():
    """
    Show effect of MTI filtering
    """
    # Create scenario
    radar = PulseDopplerRadar(num_pulses=128, samples_per_pulse=512)
    
    # Strong clutter
    for r in [500, 1000, 1500, 2000]:
        radar.add_target(range_m=r, velocity_ms=0, rcs=10.0)  # Very strong
    
    # Weak moving target
    radar.add_target(range_m=1200, velocity_ms=20, rcs=0.1)  # Weak drone
    
    radar.add_noise(snr_db=10)
    
    # Generate RDMs
    rdm_no_mti, range_axis, velocity_axis = radar.generate_rdm()
    rdm_with_mti = apply_mti(radar, rdm_no_mti)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, rdm, title in zip(axes, 
                              [rdm_no_mti, rdm_with_mti],
                              ['Without MTI', 'With MTI Filter']):
        rdm_db = 20 * np.log10(rdm + 1e-10)
        rdm_db -= np.max(rdm_db)
        
        range_bins = range_axis < 3000
        
        im = ax.imshow(
            rdm_db[:, range_bins],
            aspect='auto',
            extent=[0, 3, velocity_axis[0], velocity_axis[-1]],
            cmap='jet',
            vmin=-60,
            vmax=0,
            origin='lower'
        )
        
        ax.set_xlabel('Range (km)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(title, fontweight='bold')
        ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, color='white')
        
        plt.colorbar(im, ax=ax, label='dB')
    
    plt.tight_layout()
    plt.savefig('/Users/amlanchatterjee/mti_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Uncomment to run:
# compare_with_without_mti()

# %% [markdown]
# ---
# # 6. Exercises & Experiments
# 
# Before moving to Part 3 (Micro-Doppler), try these:
# 
# ## Exercise 1: Resolution Limits
# Create two targets at:
# - Same velocity (20 m/s) but ranges 1000m and 1001m
# - Can you resolve them? What bandwidth do you need?
# 
# ## Exercise 2: Velocity Ambiguity
# Add a target with velocity 100 m/s (faster than max unambiguous)
# - Where does it appear in the RDM?
# - This is "Doppler folding" - understand it!
# 
# ## Exercise 3: SNR Impact
# Reduce SNR from 15 dB to 5 dB, then 0 dB
# - At what point do targets disappear?
# - This motivates CFAR detection (coming later)
# 
# ## Exercise 4: CPI Length
# Change num_pulses from 128 to 64, then 32
# - How does velocity resolution change?
# - Trade-off: resolution vs update rate
# 
# ---
# 
# # Key Takeaways
# 
# ✅ **RDM = 2D FFT** of pulse-Doppler data matrix  
# ✅ **Horizontal axis** = Range (from time delay)  
# ✅ **Vertical axis** = Velocity (from Doppler shift)  
# ✅ **MTI filters** remove stationary clutter  
# ✅ **Resolution** depends on bandwidth (range) and CPI length (velocity)  
# ✅ **Ambiguities** exist - max range and max velocity are limited  
# 
# ---
# 
# ## What's Next?
# 
# You now have the core RDM generation pipeline working. In **Part 3**, we'll add:
# 
# 1. **Micro-Doppler signatures** - The rotor modulation that distinguishes drones from birds
# 2. **CFAR detection** - Automatic target detection in noise/clutter
# 3. **Target tracking** - Following targets across multiple CPIs
# 
# **Ready?** Run the examples above, play with parameters, then let me know what questions you have!
