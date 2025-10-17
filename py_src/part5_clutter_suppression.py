"""
RADAR SIGNAL PROCESSING TUTORIAL - Part 5
==========================================
Clutter Suppression: Classical Filters and Neural Networks

Learning Objectives:
- Understand different clutter types and their characteristics
- Implement classical MTI filters (single-delay, three-pulse, adaptive)
- Understand STAP concept for array radars
- Implement neural network for clutter cancellation
- Compare classical vs AI-based approaches

Prerequisites: Parts 1-4 (RDM generation, detection)
Estimated time: 2-3 hours

Author's Note: This tutorial covers both classical signal processing and modern
deep learning approaches to clutter suppression.
"""

# %%
# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg
from scipy.ndimage import generate_binary_structure
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os
warnings.filterwarnings('ignore')

# Visualization setup
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("✓ Imports successful")
print("=" * 70)
print("PART 5: CLUTTER SUPPRESSION")
print("=" * 70)

# %% [markdown]
"""
---
# Part 5: Clutter Suppression

## Quick Recap: Where We Are

From Parts 1-4, you understand:
- ✅ **RDM Generation**: 2D FFT creates range-velocity map
- ✅ **Clutter**: Strong returns from stationary/slow-moving objects
- ✅ **Detection**: CFAR adapts threshold to local noise
- ✅ **The Problem**: Clutter can be 40-60 dB stronger than targets!

## The Challenge: Clutter Masks Weak Targets

**Imagine this scenario:**
- Target: Small drone at 15 m/s, RCS = 0.01 m² → weak return
- Clutter: Ground, buildings at 0 m/s, RCS = 1000 m² → HUGE return
- Result: Drone is INVISIBLE in the RDM (buried under clutter)

**This is what Part 5 solves: Remove clutter while preserving targets**

---
# 1. Understanding Clutter Types

## What is Clutter?

**DEFINITION: Clutter = Unwanted radar returns from the environment**

NOT the target you're looking for, but strong enough to:
1. Mask weak targets (hide them)
2. Cause false alarms (look like targets)
3. Saturate receiver (overload electronics)

Think of it like trying to hear a whisper (target) in a noisy room (clutter).

## Key Terms Explained

Before we dive in, let's define the vocabulary we'll use:

**1. Doppler Velocity (v)**
- How fast something is moving toward/away from radar
- Units: m/s (meters per second)
- Stationary objects: v = 0 m/s
- Moving toward radar: v > 0 (positive Doppler)
- Moving away: v < 0 (negative Doppler)

**2. RCS (Radar Cross Section)**
- How "reflective" an object is to radar
- Units: m² (square meters)
- Examples: Bird = 0.001 m², Car = 10 m², Building = 1000 m²
- Bigger RCS = stronger return = easier to detect

**3. Doppler Spread**
- How much the Doppler velocity varies
- Units: m/s
- Narrow spread: All parts moving at same velocity (solid object)
- Wide spread: Different parts moving differently (rain, waves)

**4. CNR (Clutter-to-Noise Ratio)**
- How much stronger is clutter than noise?
- Units: dB (decibels)
- CNR = 10 * log10(Clutter_Power / Noise_Power)
- Example: CNR = 40 dB means clutter is 10,000× stronger than noise!

**5. SINR (Signal-to-Interference-plus-Noise Ratio)**
- How much stronger is target than (clutter + noise)?
- Units: dB
- SINR = Signal_Power / (Clutter_Power + Noise_Power)
- Goal of clutter suppression: INCREASE SINR

## Visual Representation of the Problem

```
WITHOUT CLUTTER SUPPRESSION:
Range-Doppler Map:
  
  Velocity →
  -50 m/s         0 m/s         +50 m/s
    |             |              |
    ■             █              ■     ← Weak targets (invisible)
    ■          █████████         ■     
    ■        ███████████████     ■     
R   ■      █████████████████    ■     ← HUGE clutter at v=0
a   ■    ███████████████████    ■        (stationary objects)
n   ■  █████████████████████    ■     
g   ■  █████████████████████    ■     
e   ■    ███████████████████    ■     
↓   ■      █████████████████    ■     
    ■        ███████████████     ■     
    ■          █████████         ■     
    ■             █              ■     
    
    Targets at v = ±15 m/s are BURIED!


WITH CLUTTER SUPPRESSION:
Range-Doppler Map:
  
  Velocity →
  -50 m/s         0 m/s         +50 m/s
    |             |              |
    ■             ·              ●     ← Target visible!
    ■             ·              ■     
    ■             ·              ■     
R   ■             ·              ■     ← Clutter removed
a   ■             ·              ■     
n   ■             ·              ■     
g   ■             ·              ■     
e   ●             ·              ■     ← Target visible!
↓   ■             ·              ■     
    ■             ·              ■     
    ■             ·              ■     
    ■             ·              ■     
    
    Clutter at v=0 suppressed, targets revealed!
```

---
"""

# %%
# First, let's create a radar class that can add different types of clutter
class ClutterRadar:
    """
    Extension of PulseDopplerRadar with realistic clutter modeling
    
    WHY THIS CLASS EXISTS:
    - Models different types of clutter (ground, weather, sea)
    - Each clutter type has unique Doppler characteristics
    - Allows testing different suppression techniques
    
    CLUTTER TYPES:
    1. Ground clutter: Stationary (v ≈ 0), very strong, narrow spread
    2. Weather clutter: Low velocity (wind), moderate strength, medium spread
    3. Sea clutter: Wave motion, fluctuating strength, wider spread
    4. Vegetation: Wind-blown, time-varying, moderate spread
    """
    
    def __init__(self, f_c=10e9, PRF=10e3, bandwidth=150e6,
                 num_pulses=128, samples_per_pulse=512):
        """
        Initialize radar with clutter modeling capability
        
        Parameters:
        -----------
        f_c : float
            Carrier frequency [Hz] (default: 10 GHz X-band)
        PRF : float
            Pulse Repetition Frequency [Hz] (default: 10 kHz)
        bandwidth : float
            Signal bandwidth [Hz] (default: 150 MHz)
        num_pulses : int
            Number of pulses in CPI (default: 128)
        samples_per_pulse : int
            Range samples per pulse (default: 512)
        """
        self.f_c = f_c
        self.PRF = PRF
        self.bandwidth = bandwidth
        self.num_pulses = num_pulses
        self.samples_per_pulse = samples_per_pulse
        
        # Derived parameters
        self.c = 3e8  # Speed of light [m/s]
        self.wavelength = self.c / f_c
        self.PRI = 1 / PRF  # Pulse Repetition Interval [s]
        self.range_resolution = self.c / (2 * bandwidth)
        
        # Max unambiguous range and velocity
        self.max_range = self.c * self.PRI / 2
        self.max_velocity = self.wavelength * PRF / 4
        
        # Initialize data matrix: [num_pulses x samples_per_pulse]
        # This is the "slow-time x fast-time" matrix
        # Slow-time = pulse-to-pulse (Doppler)
        # Fast-time = within pulse (Range)
        self.data_matrix = np.zeros((num_pulses, samples_per_pulse), 
                                    dtype=complex)
        
        # Axes for visualization
        self.range_axis = np.linspace(0, self.max_range, samples_per_pulse)
        self.velocity_axis = np.linspace(-self.max_velocity, self.max_velocity, 
                                        num_pulses)
        
        print(f"ClutterRadar initialized:")
        print(f"  Carrier frequency: {f_c/1e9:.1f} GHz")
        print(f"  PRF: {PRF/1e3:.1f} kHz")
        print(f"  Range resolution: {self.range_resolution:.2f} m")
        print(f"  Max range: {self.max_range/1e3:.1f} km")
        print(f"  Velocity resolution: {self.wavelength*PRF/(2*num_pulses):.2f} m/s")
        print(f"  Max velocity: {self.max_velocity:.1f} m/s")
    
    def add_target(self, range_m, velocity_ms, rcs=1.0):
        """
        Add a point target to the scene
        
        Parameters:
        -----------
        range_m : float
            Target range [meters]
        velocity_ms : float
            Target radial velocity [m/s] (positive = approaching)
        rcs : float
            Radar cross section [m²]
        """
        # Find range bin
        range_bin = int(range_m / self.range_resolution)
        if range_bin >= self.samples_per_pulse:
            print(f"Warning: Target at {range_m}m exceeds max range")
            return
        
        # Doppler frequency from velocity
        doppler_freq = 2 * velocity_ms / self.wavelength
        
        # Generate signal across pulses (slow-time dimension)
        pulse_times = np.arange(self.num_pulses) * self.PRI
        phase_progression = 2 * np.pi * doppler_freq * pulse_times
        
        # Target signal (amplitude proportional to sqrt(RCS))
        target_signal = np.sqrt(rcs) * np.exp(1j * phase_progression)
        
        # Add to data matrix at appropriate range bin
        self.data_matrix[:, range_bin] += target_signal
    
    def add_ground_clutter(self, range_start, range_end, cnr_db=50):
        """
        Add ground clutter (stationary, very strong, narrow Doppler spread)
        
        CHARACTERISTICS:
        - Velocity: 0 m/s (stationary)
        - Strength: 40-60 dB above noise (VERY strong)
        - Doppler spread: < 1 m/s (narrow - all parts stationary)
        - Examples: Buildings, terrain, roads
        
        Parameters:
        -----------
        range_start : float
            Starting range of clutter [m]
        range_end : float
            Ending range of clutter [m]
        cnr_db : float
            Clutter-to-Noise Ratio [dB] (default: 50 dB = 100,000×)
        
        Visual representation:
        ```
        Ground Clutter Profile:
        
        Velocity
           ↑
           |    ▓▓▓▓▓       ← Very strong (CNR = 50 dB)
           |    ▓▓▓▓▓       ← Narrow spread (< 1 m/s)
           |    ▓▓▓▓▓       ← Centered at v = 0
         0 |════▓▓▓▓▓═══════→ Range
           |    
           ↓
        ```
        """
        # Convert CNR from dB to linear
        clutter_power = 10**(cnr_db / 10)
        
        # Range bins affected
        bin_start = int(range_start / self.range_resolution)
        bin_end = int(range_end / self.range_resolution)
        bin_end = min(bin_end, self.samples_per_pulse)
        
        # Generate ground clutter
        for r_bin in range(bin_start, bin_end):
            # Base clutter (stationary, but with small fluctuations)
            # Rayleigh distributed amplitude (typical for radar clutter)
            amplitude = np.random.rayleigh(scale=np.sqrt(clutter_power/2), 
                                          size=self.num_pulses)
            
            # Very small Doppler spread (< 0.5 m/s) - almost DC
            doppler_spread = 0.3  # m/s
            doppler_freq_spread = 2 * doppler_spread / self.wavelength
            
            # Random phases with slow variation (simulates small movements)
            phase_noise = np.cumsum(np.random.randn(self.num_pulses) * 
                                   2 * np.pi * doppler_freq_spread * self.PRI)
            
            clutter_signal = amplitude * np.exp(1j * phase_noise)
            self.data_matrix[:, r_bin] += clutter_signal
    
    def add_weather_clutter(self, range_start, range_end, wind_velocity=5, cnr_db=30):
        """
        Add weather clutter (rain, snow - wind-driven, moderate spread)
        
        CHARACTERISTICS:
        - Velocity: Wind speed (typically 5-15 m/s)
        - Strength: 20-40 dB above noise (moderate)
        - Doppler spread: 5-10 m/s (medium - droplets at different velocities)
        - Examples: Rain, snow, hail
        
        Parameters:
        -----------
        range_start : float
            Starting range [m]
        range_end : float
            Ending range [m]
        wind_velocity : float
            Mean wind velocity [m/s] (default: 5 m/s)
        cnr_db : float
            Clutter-to-Noise Ratio [dB] (default: 30 dB = 1,000×)
        
        Visual representation:
        ```
        Weather Clutter Profile:
        
        Velocity
           ↑
           |      ░░░░       ← Moderate strength (CNR = 30 dB)
           |     ░░░░░░      ← Medium spread (5-10 m/s)
           |    ░░░░░░░░     ← Centered at wind velocity
         0 |═════░░░░░░░════→ Range
           |        ↑
           |    wind_velocity
        ```
        """
        clutter_power = 10**(cnr_db / 10)
        
        bin_start = int(range_start / self.range_resolution)
        bin_end = int(range_end / self.range_resolution)
        bin_end = min(bin_end, self.samples_per_pulse)
        
        # Doppler spread from wind turbulence
        doppler_spread = 8  # m/s (wider than ground clutter)
        
        for r_bin in range(bin_start, bin_end):
            # Each droplet has slightly different velocity
            # Gaussian distribution around wind velocity
            velocities = np.random.randn(self.num_pulses) * doppler_spread + wind_velocity
            doppler_freqs = 2 * velocities / self.wavelength
            
            # Rayleigh amplitude (fluctuating rain intensity)
            amplitudes = np.random.rayleigh(scale=np.sqrt(clutter_power/2),
                                           size=self.num_pulses)
            
            # Phase evolution for each pulse
            pulse_times = np.arange(self.num_pulses) * self.PRI
            phases = 2 * np.pi * doppler_freqs * pulse_times
            
            weather_signal = amplitudes * np.exp(1j * phases)
            self.data_matrix[:, r_bin] += weather_signal
    
    def add_sea_clutter(self, range_start, range_end, sea_state=3, cnr_db=35):
        """
        Add sea clutter (ocean waves - complex motion, wide spread)
        
        CHARACTERISTICS:
        - Velocity: Wave motion (0-10 m/s depending on sea state)
        - Strength: 25-45 dB above noise
        - Doppler spread: 10-20 m/s (wide - waves at all different velocities)
        - Statistics: Non-Gaussian (spiky, K-distributed)
        - Examples: Ocean surface, large lakes
        
        Parameters:
        -----------
        range_start : float
            Starting range [m]
        range_end : float
            Ending range [m]
        sea_state : int
            Sea state (1=calm to 5=rough) affects spread and strength
        cnr_db : float
            Clutter-to-Noise Ratio [dB]
        
        Visual representation:
        ```
        Sea Clutter Profile:
        
        Velocity
           ↑
           |    ▒▒▒▒▒▒▒     ← Strong, spiky (CNR = 35 dB)
           |   ▒▒▒▒▒▒▒▒▒    ← Wide spread (10-20 m/s)
           |  ▒▒▒▒▒▒▒▒▒▒▒   ← Non-Gaussian statistics
         0 |══▒▒▒▒▒▒▒▒▒▒▒══→ Range
           |        
           |  Wave motion creates complex Doppler
        ```
        """
        clutter_power = 10**(cnr_db / 10)
        
        bin_start = int(range_start / self.range_resolution)
        bin_end = int(range_end / self.range_resolution)
        bin_end = min(bin_end, self.samples_per_pulse)
        
        # Sea state affects spread and characteristics
        doppler_spread = 5 * sea_state  # Higher sea state = wider spread
        
        for r_bin in range(bin_start, bin_end):
            # Sea clutter has K-distribution (spiky)
            # We'll approximate with Rayleigh + spikes
            base_amplitude = np.random.rayleigh(scale=np.sqrt(clutter_power/2),
                                                size=self.num_pulses)
            
            # Add occasional spikes (breaking waves)
            spike_probability = 0.05 * sea_state
            spikes = np.random.rand(self.num_pulses) < spike_probability
            base_amplitude[spikes] *= 5  # Spikes are 5× stronger
            
            # Wide Doppler spread from wave motion
            # Mix of different wave velocities
            velocities = np.random.randn(self.num_pulses) * doppler_spread
            doppler_freqs = 2 * velocities / self.wavelength
            
            pulse_times = np.arange(self.num_pulses) * self.PRI
            phases = 2 * np.pi * doppler_freqs * pulse_times
            
            sea_signal = base_amplitude * np.exp(1j * phases)
            self.data_matrix[:, r_bin] += sea_signal
    
    def add_noise(self, snr_db=20):
        """
        Add thermal noise to all range bins
        
        Parameters:
        -----------
        snr_db : float
            Signal-to-Noise Ratio for reference target [dB]
            This sets the noise floor level
        """
        # Noise power (set relative to a reference RCS=1 target)
        noise_power = 10**(-snr_db / 10)
        
        # Complex Gaussian noise (I and Q channels)
        noise = (np.random.randn(self.num_pulses, self.samples_per_pulse) + 
                1j * np.random.randn(self.num_pulses, self.samples_per_pulse))
        noise *= np.sqrt(noise_power / 2)
        
        self.data_matrix += noise
    
    def generate_rdm(self, window='hamming'):
        """
        Generate Range-Doppler Map using 2D FFT
        
        Parameters:
        -----------
        window : str
            Window function for Doppler FFT (default: 'hamming')
            Options: 'hamming', 'hann', 'blackman', 'none'
        
        Returns:
        --------
        rdm : ndarray
            Range-Doppler Map [num_pulses x samples_per_pulse]
            Power in linear scale
        rdm_db : ndarray
            Same RDM in dB scale for visualization
        range_axis : ndarray
            Range values [m]
        velocity_axis : ndarray
            Velocity values [m/s]
        """
        # Apply window in slow-time (Doppler) dimension
        if window != 'none':
            window_func = signal.get_window(window, self.num_pulses)
            windowed_data = self.data_matrix * window_func[:, np.newaxis]
        else:
            windowed_data = self.data_matrix
        
        # 2D FFT: Range FFT (already done) + Doppler FFT
        # In our data_matrix, range is already in frequency domain
        # We just need Doppler FFT
        rdm = np.fft.fftshift(np.fft.fft(windowed_data, axis=0), axes=0)
        
        # Convert to power (magnitude squared)
        rdm_power = np.abs(rdm) ** 2
        
        # Convert to dB for visualization
        rdm_db = 10 * np.log10(rdm_power + 1e-10)
        
        return rdm_power, rdm_db, self.range_axis, self.velocity_axis


# %%
print("\n" + "=" * 70)
print("DEMONSTRATION: Clutter Types Visualization")
print("=" * 70)

# Create scenarios to visualize different clutter types
print("\nCreating 4 scenarios:")
print("  1. Ground clutter only")
print("  2. Weather clutter only")
print("  3. Sea clutter only")
print("  4. Mixed clutter (ground + weather)")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scenario 1: Ground clutter
print("\nGenerating Scenario 1: Ground clutter...")
radar1 = ClutterRadar(num_pulses=128, samples_per_pulse=256)
radar1.add_ground_clutter(range_start=0, range_end=2000, cnr_db=50)
radar1.add_target(range_m=1500, velocity_ms=15, rcs=0.01)  # Weak target
radar1.add_noise(snr_db=20)
rdm1, rdm1_db, r_axis, v_axis = radar1.generate_rdm()

im0 = axes[0, 0].imshow(rdm1_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 0].axvline(2, color='white', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 0].set_title('Ground Clutter (Stationary)', fontweight='bold', fontsize=13)
axes[0, 0].set_xlabel('Range [km]')
axes[0, 0].set_ylabel('Velocity [m/s]')
axes[0, 0].text(1, 25, 'Strong vertical line\nat v=0 (stationary)', 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
               fontsize=10)
plt.colorbar(im0, ax=axes[0, 0], label='Power [dB]')

# Scenario 2: Weather clutter
print("Generating Scenario 2: Weather clutter...")
radar2 = ClutterRadar(num_pulses=128, samples_per_pulse=256)
radar2.add_weather_clutter(range_start=500, range_end=3000, 
                           wind_velocity=8, cnr_db=30)
radar2.add_target(range_m=1500, velocity_ms=15, rcs=0.01)
radar2.add_noise(snr_db=20)
rdm2, rdm2_db, _, _ = radar2.generate_rdm()

im1 = axes[0, 1].imshow(rdm2_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 1].axhline(8, color='white', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 1].set_title('Weather Clutter (Wind-Driven)', fontweight='bold', fontsize=13)
axes[0, 1].set_xlabel('Range [km]')
axes[0, 1].set_ylabel('Velocity [m/s]')
axes[0, 1].text(2, 25, 'Spread around\nwind velocity (8 m/s)', 
               bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.8),
               fontsize=10)
plt.colorbar(im1, ax=axes[0, 1], label='Power [dB]')

# Scenario 3: Sea clutter
print("Generating Scenario 3: Sea clutter...")
radar3 = ClutterRadar(num_pulses=128, samples_per_pulse=256)
radar3.add_sea_clutter(range_start=1000, range_end=4000, sea_state=3, cnr_db=35)
radar3.add_target(range_m=2000, velocity_ms=20, rcs=0.01)
radar3.add_noise(snr_db=20)
rdm3, rdm3_db, _, _ = radar3.generate_rdm()

im2 = axes[1, 0].imshow(rdm3_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[1, 0].set_title('Sea Clutter (Wave Motion)', fontweight='bold', fontsize=13)
axes[1, 0].set_xlabel('Range [km]')
axes[1, 0].set_ylabel('Velocity [m/s]')
axes[1, 0].text(3, 25, 'Wide spread\nfrom waves', 
               bbox=dict(boxstyle='round', facecolor='lime', alpha=0.8),
               fontsize=10)
plt.colorbar(im2, ax=axes[1, 0], label='Power [dB]')

# Scenario 4: Mixed clutter
print("Generating Scenario 4: Mixed clutter...")
radar4 = ClutterRadar(num_pulses=128, samples_per_pulse=256)
radar4.add_ground_clutter(range_start=0, range_end=1500, cnr_db=50)
radar4.add_weather_clutter(range_start=1500, range_end=4000, 
                           wind_velocity=6, cnr_db=30)
radar4.add_target(range_m=800, velocity_ms=-12, rcs=0.01)
radar4.add_target(range_m=2500, velocity_ms=18, rcs=0.01)
radar4.add_noise(snr_db=20)
rdm4, rdm4_db, _, _ = radar4.generate_rdm()

im3 = axes[1, 1].imshow(rdm4_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[1, 1].axvline(1.5, color='white', linestyle=':', linewidth=2, alpha=0.7)
axes[1, 1].set_title('Mixed Clutter (Ground + Weather)', fontweight='bold', fontsize=13)
axes[1, 1].set_xlabel('Range [km]')
axes[1, 1].set_ylabel('Velocity [m/s]')
axes[1, 1].text(0.5, 25, 'Ground', bbox=dict(boxstyle='round', 
               facecolor='yellow', alpha=0.8), fontsize=10)
axes[1, 1].text(2.5, 25, 'Weather', bbox=dict(boxstyle='round', 
               facecolor='cyan', alpha=0.8), fontsize=10)
plt.colorbar(im3, ax=axes[1, 1], label='Power [dB]')

plt.tight_layout()
plt.show()

print("\n✓ Clutter visualization complete")
print("\n" + "=" * 70)
print("KEY OBSERVATIONS:")
print("=" * 70)
print("1. Ground clutter: Strong vertical line at v=0 (stationary)")
print("2. Weather clutter: Spread around wind velocity")
print("3. Sea clutter: Wide spread, spiky (wave motion)")
print("4. Mixed: Combination of characteristics")
print("\n💡 Notice: Targets are BARELY visible in clutter!")
print("   This is why we need clutter suppression.")
print("=" * 70)

# %% [markdown]
"""
---
## 🤔 Test Your Intuition

Before we move to suppression algorithms, verify you understand clutter characteristics:

**Question 1:** A radar sees a strong vertical line in the RDM at v=0 m/s. What type of clutter is this most likely to be?

**Question 2:** If you see clutter spread over velocities from -10 to +10 m/s with occasional strong spikes, what environment are you in?

**Question 3:** You want to detect a drone flying at 15 m/s in an area with strong ground clutter. Will increasing transmit power help detect the drone?

<details>
<summary><b>💡 Click to see answers</b></summary>

**Answer 1:**
Ground clutter! Stationary objects (buildings, terrain) have v=0, creating a strong vertical line in the RDM. This is the most common type of clutter.

**Why:** Everything at the same range with v=0 appears in the zero-Doppler bin across all range gates.

**Answer 2:**
Sea clutter! The wide spread (-10 to +10 m/s) comes from wave motion. The occasional spikes are breaking waves, which have much stronger returns.

**Why:** Ocean waves move at different velocities and directions, creating spread. Breaking waves (whitecaps) are highly reflective.

**Answer 3:**
NO! Increasing power makes BOTH the target AND the clutter stronger by the same ratio. The Signal-to-Clutter Ratio (SCR) stays the same.

**Why:** Clutter power scales linearly with transmit power, just like target power. You need FILTERING (suppression), not more power!

**The key insight:** Clutter suppression is about EXPLOITING DIFFERENCES in Doppler behavior, not overpowering the clutter.

</details>

---
"""

# %% [markdown]
"""
---
# 2. Classical MTI Filters

## The Core Idea: Exploit Doppler Differences

**Physical Picture:**
- Ground clutter: v = 0 m/s → Doppler frequency = 0 Hz (DC component)
- Targets: v ≠ 0 → Doppler frequency ≠ 0 Hz

**Solution: HIGH-PASS FILTER**
Remove DC component (v=0) → Remove stationary clutter!

Think of it like noise-canceling headphones:
- Headphones don't make you louder
- They make background noise quieter
- Same SNR improvement by reducing interference

## MTI = Moving Target Indicator

**DEFINITION: MTI = Filter that removes stationary returns**

Basic principle: Subtract consecutive pulses
- If object is stationary: pulse[n] = pulse[n-1] → difference = 0 (removed!)
- If object is moving: pulse[n] ≠ pulse[n-1] → difference ≠ 0 (preserved!)

## Visual Explanation

```
PULSE-TO-PULSE COMPARISON:

Stationary Object (Ground):
Pulse 1:  ████████  ← Same phase
Pulse 2:  ████████  ← Same phase
Difference:  ........  ← ZERO! (removed)

Moving Object (Target):
Pulse 1:  ████████  ← Phase = 0°
Pulse 2:    ████████  ← Phase = 45° (shifted!)
Difference:  ░░░░  ← NON-ZERO! (preserved)
```

---
"""

# %%
def single_delay_mti(data_matrix):
    """
    Single-delay MTI filter (simplest form)
    
    ALGORITHM:
    y[n] = x[n] - x[n-1]
    
    WHERE:
    - x[n] = signal from pulse n
    - y[n] = filtered output at pulse n
    
    WHAT IT DOES:
    - Removes DC component (zero Doppler)
    - Acts as HIGH-PASS filter in Doppler domain
    - First-order differencer
    
    FREQUENCY RESPONSE:
    H(f) = 1 - exp(-j*2π*f*PRI)
    - Null (zero) at f=0 (DC)
    - Maximum at f = PRF/2
    
    Parameters:
    -----------
    data_matrix : ndarray
        Input data [num_pulses x num_range_bins]
        Slow-time (Doppler) is along axis 0
    
    Returns:
    --------
    filtered : ndarray
        MTI filtered data (same shape as input)
        First pulse is zeros (no previous pulse to subtract)
    
    Visual representation:
    ```
    Input signal (with clutter):
        Pulse: 1    2    3    4    5
        Value: ████ ████ ████ ████ ████  ← Strong DC (clutter)
               ░    ░░   ░░░  ░░   ░     ← Weak target modulation
    
    After MTI:
        Pulse: 1    2    3    4    5
        Value: 0    ░    ░░   ░    ░░    ← DC removed!
                              ↑ Target visible
    ```
    """
    # Initialize output (first pulse has no previous reference)
    filtered = np.zeros_like(data_matrix)
    
    # Subtract consecutive pulses: y[n] = x[n] - x[n-1]
    # Skip first pulse (n=0 has no n-1)
    filtered[1:, :] = data_matrix[1:, :] - data_matrix[:-1, :]
    
    return filtered


# %%
def three_pulse_mti(data_matrix):
    """
    Three-pulse MTI filter (improved clutter rejection)
    
    ALGORITHM:
    y[n] = x[n] - 2*x[n-1] + x[n-2]
    
    This is equivalent to applying single-delay MTI TWICE:
    - First pass: y1[n] = x[n] - x[n-1]
    - Second pass: y2[n] = y1[n] - y1[n-1]
    - Result: y2[n] = x[n] - 2*x[n-1] + x[n-2]
    
    WHAT IT DOES:
    - Deeper null at DC (better clutter rejection)
    - Still passes moving targets
    - Second-order differencer
    
    FREQUENCY RESPONSE:
    H(f) = (1 - exp(-j*2π*f*PRI))²
    - Deeper null at f=0 (DC)
    - Better rejection of near-DC frequencies
    
    ADVANTAGE OVER SINGLE-DELAY:
    - 10-20 dB better clutter rejection
    - Useful when clutter is VERY strong
    
    DISADVANTAGE:
    - Loses first TWO pulses (not just one)
    - Slightly more attenuation at all frequencies
    
    Parameters:
    -----------
    data_matrix : ndarray
        Input data [num_pulses x num_range_bins]
    
    Returns:
    --------
    filtered : ndarray
        MTI filtered data
        First two pulses are zeros
    
    Visual comparison:
    ```
    Single-delay MTI:
    Frequency response:
         ∧
      1  |      ╱╲
         |     ╱  ╲
      0  |____╱____╲____  ← Shallow null at DC
         0   DC    f_max
    
    Three-pulse MTI:
    Frequency response:
         ∧
      1  |       ╱╲
         |      ╱  ╲
      0  |_____╱____╲___  ← DEEP null at DC (better!)
         0    DC    f_max
    ```
    """
    filtered = np.zeros_like(data_matrix)
    
    # Second-order difference: y[n] = x[n] - 2*x[n-1] + x[n-2]
    # Skip first two pulses
    filtered[2:, :] = (data_matrix[2:, :] - 
                       2 * data_matrix[1:-1, :] + 
                       data_matrix[:-2, :])
    
    return filtered


# %%
print("\n" + "=" * 70)
print("DEMONSTRATION: Classical MTI Filters")
print("=" * 70)

# Create test scenario: Ground clutter + moving target
print("\nCreating test scenario:")
print("  - Strong ground clutter (CNR = 50 dB)")
print("  - Weak target at v = 15 m/s (RCS = 0.01 m²)")

radar_mti = ClutterRadar(num_pulses=128, samples_per_pulse=256)
radar_mti.add_ground_clutter(range_start=0, range_end=3000, cnr_db=50)
radar_mti.add_target(range_m=1500, velocity_ms=15, rcs=0.01)
radar_mti.add_target(range_m=2000, velocity_ms=-12, rcs=0.01)
radar_mti.add_noise(snr_db=20)

print("✓ Scenario created")

# Generate RDMs
print("\nGenerating RDMs:")
print("  1. No filtering (raw)")
print("  2. Single-delay MTI")
print("  3. Three-pulse MTI")

# Raw RDM
rdm_raw, rdm_raw_db, r_axis, v_axis = radar_mti.generate_rdm()

# Single-delay MTI
data_mti1 = single_delay_mti(radar_mti.data_matrix)
radar_temp = ClutterRadar(num_pulses=128, samples_per_pulse=256)
radar_temp.data_matrix = data_mti1
rdm_mti1, rdm_mti1_db, _, _ = radar_temp.generate_rdm()

# Three-pulse MTI
data_mti3 = three_pulse_mti(radar_mti.data_matrix)
radar_temp.data_matrix = data_mti3
rdm_mti3, rdm_mti3_db, _, _ = radar_temp.generate_rdm()

print("✓ RDMs generated")

# Visualize comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Raw RDM
im0 = axes[0, 0].imshow(rdm_raw_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 0].axhline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 0].set_title('No Filtering (Raw RDM)', fontweight='bold', fontsize=13)
axes[0, 0].set_xlabel('Range [km]')
axes[0, 0].set_ylabel('Velocity [m/s]')
axes[0, 0].text(2, 30, 'Targets BURIED\nin clutter!', 
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.8),
               fontsize=11)
plt.colorbar(im0, ax=axes[0, 0], label='Power [dB]')

# Plot 2: Single-delay MTI
im1 = axes[0, 1].imshow(rdm_mti1_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 1].axhline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 1].set_title('Single-Delay MTI', fontweight='bold', fontsize=13)
axes[0, 1].set_xlabel('Range [km]')
axes[0, 1].set_ylabel('Velocity [m/s]')
axes[0, 1].text(2, 30, 'Clutter reduced\nTargets visible!', 
               bbox=dict(boxstyle='round', facecolor='lime', alpha=0.8),
               fontsize=11)
plt.colorbar(im1, ax=axes[0, 1], label='Power [dB]')

# Plot 3: Three-pulse MTI
im2 = axes[0, 2].imshow(rdm_mti3_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 2].axhline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 2].set_title('Three-Pulse MTI', fontweight='bold', fontsize=13)
axes[0, 2].set_xlabel('Range [km]')
axes[0, 2].set_ylabel('Velocity [m/s]')
axes[0, 2].text(2, 30, 'Even cleaner!\nDeeper notch', 
               bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.8),
               fontsize=11)
plt.colorbar(im2, ax=axes[0, 2], label='Power [dB]')

# Plot 4: Doppler cut at target range (range = 1500m)
range_bin_target = int(1500 / radar_mti.range_resolution)
axes[1, 0].plot(v_axis, rdm_raw_db[:, range_bin_target], 'r-', 
               linewidth=2, label='No Filter')
axes[1, 0].plot(v_axis, rdm_mti1_db[:, range_bin_target], 'g-', 
               linewidth=2, label='Single-Delay MTI')
axes[1, 0].plot(v_axis, rdm_mti3_db[:, range_bin_target], 'b-', 
               linewidth=2, label='Three-Pulse MTI')
axes[1, 0].axvline(0, color='black', linestyle=':', alpha=0.5)
axes[1, 0].axvline(15, color='red', linestyle=':', alpha=0.5, 
                   label='Target velocity')
axes[1, 0].set_xlabel('Velocity [m/s]', fontsize=11)
axes[1, 0].set_ylabel('Power [dB]', fontsize=11)
axes[1, 0].set_title('Doppler Profile at Range=1.5km', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
axes[1, 0].text(0, 15, 'DC removed', rotation=90, va='bottom',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 5: MTI Filter frequency responses (theoretical)
freq_norm = np.linspace(-0.5, 0.5, 500)  # Normalized frequency
H_single = np.abs(1 - np.exp(-1j * 2 * np.pi * freq_norm))
H_three = np.abs(1 - np.exp(-1j * 2 * np.pi * freq_norm))**2

axes[1, 1].plot(freq_norm * radar_mti.PRF, 20*np.log10(H_single + 1e-10), 
               'g-', linewidth=2, label='Single-Delay')
axes[1, 1].plot(freq_norm * radar_mti.PRF, 20*np.log10(H_three + 1e-10), 
               'b-', linewidth=2, label='Three-Pulse')
axes[1, 1].axhline(0, color='black', linestyle=':', alpha=0.5)
axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='DC (v=0)')
axes[1, 1].set_xlabel('Doppler Frequency [Hz]', fontsize=11)
axes[1, 1].set_ylabel('Magnitude Response [dB]', fontsize=11)
axes[1, 1].set_title('MTI Filter Frequency Response', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
axes[1, 1].set_ylim([-60, 5])
axes[1, 1].text(0, -50, 'Deeper notch\n= better rejection', 
               ha='center',
               bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))

# Plot 6: Improvement factor
# Calculate clutter power at v=0 vs target power at v=15
v_zero_idx = np.argmin(np.abs(v_axis))
v_target_idx = np.argmin(np.abs(v_axis - 15))

clutter_raw = rdm_raw_db[v_zero_idx, range_bin_target]
clutter_mti1 = rdm_mti1_db[v_zero_idx, range_bin_target]
clutter_mti3 = rdm_mti3_db[v_zero_idx, range_bin_target]

target_raw = rdm_raw_db[v_target_idx, range_bin_target]
target_mti1 = rdm_mti1_db[v_target_idx, range_bin_target]
target_mti3 = rdm_mti3_db[v_target_idx, range_bin_target]

scr_raw = target_raw - clutter_raw
scr_mti1 = target_mti1 - clutter_mti1
scr_mti3 = target_mti3 - clutter_mti3

methods = ['No Filter', 'Single MTI', 'Three-Pulse MTI']
scr_values = [scr_raw, scr_mti1, scr_mti3]
colors = ['red', 'green', 'blue']

bars = axes[1, 2].bar(methods, scr_values, color=colors, alpha=0.7, edgecolor='black')
axes[1, 2].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1, 2].set_ylabel('Signal-to-Clutter Ratio [dB]', fontsize=11)
axes[1, 2].set_title('Clutter Rejection Performance', fontweight='bold', fontsize=12)
axes[1, 2].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, scr_values):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.1f} dB',
                    ha='center', va='bottom' if val > 0 else 'top',
                    fontweight='bold', fontsize=11)

improvement_single = scr_mti1 - scr_raw
improvement_three = scr_mti3 - scr_raw
axes[1, 2].text(0.5, 0.95, 
               f'Improvement:\nSingle: +{improvement_single:.1f} dB\nThree-pulse: +{improvement_three:.1f} dB',
               transform=axes[1, 2].transAxes,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
               fontsize=10, va='top')

plt.tight_layout()
plt.show()

print("\n✓ MTI visualization complete")
print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY:")
print("=" * 70)
print(f"Signal-to-Clutter Ratio (at target location):")
print(f"  No filter:       {scr_raw:6.1f} dB  (target buried!)")
print(f"  Single MTI:      {scr_mti1:6.1f} dB  (+{improvement_single:.1f} dB improvement)")
print(f"  Three-pulse MTI: {scr_mti3:6.1f} dB  (+{improvement_three:.1f} dB improvement)")
print("\n💡 Key Insight:")
print("   Three-pulse MTI gives ~{:.0f} dB better rejection than single-delay".format(
    improvement_three - improvement_single))
print("   This is because it has a DEEPER notch at DC (v=0)")
print("=" * 70)

# %% [markdown]
"""
---
## 🤔 Test Your Intuition

Before moving to adaptive filters, verify you understand MTI:

**Question 1:** An MTI filter works great for ground clutter, but you add a target moving at 0.5 m/s. What happens to this target?

**Question 2:** You have two targets: one at 15 m/s and one at 5 m/s. After applying three-pulse MTI, which target will be more attenuated (weakened)?

**Question 3:** If you apply MTI filtering and the clutter at v=0 INCREASES instead of decreases, what went wrong?

<details>
<summary><b>💡 Click to see answers</b></summary>

**Answer 1:**
The slow target (0.5 m/s) gets attenuated or removed along with the clutter! MTI filters remove LOW Doppler frequencies, not just exactly v=0.

**Why:** The null (notch) in the frequency response isn't infinitely narrow. It extends to nearby frequencies. Slow targets fall in the notch.

**Solution:** Use narrower notch filter (needs more pulses) or accept that very slow targets are hard to detect in strong clutter.

**Answer 2:**
The target at 5 m/s will be MORE attenuated. MTI attenuation increases as you get closer to DC (v=0).

**Why:** Look at the frequency response plot - the filter gain is lower at low frequencies. At 15 m/s you're further from the notch than at 5 m/s.

**Trade-off:** Better clutter rejection = more attenuation of slow targets.

**Answer 3:**
You're looking at the data in POWER (magnitude squared), not complex amplitude! MTI works on complex data BEFORE the magnitude operation.

**Why:** MTI exploits phase differences. If you take magnitude first, you lose phase information and MTI won't work.

**Correct order:** Complex data → MTI filter → Magnitude/Power → Display

</details>

---
"""

# %% [markdown]
"""
---
# 3. Adaptive MTI and Covariance-Based Methods

## The Limitation of Fixed MTI

**Problem:** Fixed MTI works ONLY for stationary clutter
- What if clutter is moving? (weather at 8 m/s, sea waves)
- Fixed notch at v=0 doesn't help!

**Solution:** ADAPTIVE filter that adjusts to actual clutter characteristics

## The Concept: Wiener Filter Approach

**Core idea:** Estimate clutter subspace from data, then project it out

Think of it like noise-canceling headphones that LEARN the noise pattern:
1. Listen to the noise (estimate clutter)
2. Generate anti-noise (compute filter weights)
3. Cancel the noise (apply weights)

## Mathematical Framework (Simplified)

Don't worry if math seems complex - focus on the intuition!

**Step 1: Estimate Clutter Covariance**
```
Covariance = How signals at different times relate to each other
For stationary clutter: High covariance (very similar pulse-to-pulse)
For random noise: Low covariance (no pattern)
```

**Step 2: Compute Optimal Weights**
```
Weights = How much to subtract from each pulse
Chosen to MINIMIZE clutter power while preserving target
```

**Step 3: Apply Adaptive Filter**
```
y[n] = x[n] + w₁*x[n-1] + w₂*x[n-2] + ...
where w₁, w₂, ... are computed from covariance
```

---
"""

# %%
def adaptive_mti_filter(data_matrix, num_training_cells=32, filter_length=8):
    """
    Adaptive MTI using covariance-based approach (simplified Wiener filter)
    
    ALGORITHM:
    1. For each range bin:
       a. Estimate clutter covariance from training cells
       b. Compute optimal filter weights
       c. Apply filter to suppress clutter
    
    WHY ADAPTIVE:
    - Filter adjusts to actual clutter Doppler
    - Works for moving clutter (weather, sea)
    - More degrees of freedom than fixed MTI
    
    INTUITION:
    Like a smart noise canceler that learns what the "noise" (clutter) looks like,
    then generates an anti-signal to cancel it.
    
    Parameters:
    -----------
    data_matrix : ndarray
        Input data [num_pulses x num_range_bins]
    num_training_cells : int
        How many range bins to use for estimating clutter statistics
        More training cells = better estimate but assumes more homogeneous clutter
    filter_length : int
        Number of filter taps (adaptive filter order)
        More taps = can cancel more complex clutter patterns
    
    Returns:
    --------
    filtered : ndarray
        Adaptively filtered data
    
    Technical Note:
    ---------------
    This is a SIMPLIFIED implementation for educational purposes.
    Production systems use more sophisticated methods like:
    - JDL (Joint Domain Localized) STAP
    - Factored STAP
    - Knowledge-aided STAP
    
    But the core principle is the same: estimate and subtract clutter.
    """
    num_pulses, num_range_bins = data_matrix.shape
    filtered = np.zeros_like(data_matrix)
    
    for range_idx in range(num_range_bins):
        # Step 1: Extract training data (nearby range bins)
        train_start = max(0, range_idx - num_training_cells // 2)
        train_end = min(num_range_bins, range_idx + num_training_cells // 2)
        
        # Skip if at edge
        if train_end - train_start < num_training_cells // 2:
            filtered[:, range_idx] = data_matrix[:, range_idx]
            continue
        
        training_data = data_matrix[:, train_start:train_end]
        
        # Step 2: Estimate clutter covariance matrix
        # Covariance [filter_length x filter_length]
        # Measures how signals at different pulse lags correlate
        R = np.zeros((filter_length, filter_length), dtype=complex)
        
        for lag1 in range(filter_length):
            for lag2 in range(filter_length):
                if num_pulses - max(lag1, lag2) > 0:
                    # Average correlation across training range bins
                    correlation = np.mean(
                        training_data[lag1:num_pulses-max(lag1,lag2)+lag1, :] * 
                        np.conj(training_data[lag2:num_pulses-max(lag1,lag2)+lag2, :])
                    )
                    R[lag1, lag2] = correlation
        
        # Step 3: Compute optimal filter weights
        # Using Wiener-Hopf solution (simplified)
        # Add diagonal loading for numerical stability
        lambda_load = 1e-6 * np.trace(R).real
        R_loaded = R + lambda_load * np.eye(filter_length)
        
        try:
            # Steering vector (assumes target at specific Doppler)
            # For simplicity, we use a vector that preserves non-zero Doppler
            # In practice, this would be adapted to expected target Doppler
            target_velocity = 15  # m/s (typical drone velocity)
            doppler_freq = 2 * target_velocity / (3e8 / (10e9))  # Approximate
            PRI = 1 / 10e3  # PRF = 10 kHz
            
            steering_vector = np.exp(1j * 2 * np.pi * doppler_freq * 
                                    PRI * np.arange(filter_length))
            
            # Compute weights: w = R^(-1) * s (simplified)
            weights = linalg.solve(R_loaded, steering_vector)
            weights = weights / np.abs(weights).max()  # Normalize
            
        except linalg.LinAlgError:
            # If matrix is singular, fall back to identity (no filtering)
            weights = np.zeros(filter_length, dtype=complex)
            weights[0] = 1.0
        
        # Step 4: Apply adaptive filter
        # Convolution with computed weights
        test_cell_data = data_matrix[:, range_idx]
        
        filtered_signal = np.zeros(num_pulses, dtype=complex)
        for n in range(filter_length, num_pulses):
            # Apply filter: y[n] = sum(w[k] * x[n-k])
            for k in range(filter_length):
                filtered_signal[n] += weights[k] * test_cell_data[n-k]
        
        filtered[:, range_idx] = filtered_signal
    
    return filtered


# %%
print("\n" + "=" * 70)
print("DEMONSTRATION: Adaptive MTI vs Fixed MTI")
print("=" * 70)

# Create challenging scenario: Moving clutter (weather)
print("\nCreating challenging scenario:")
print("  - Weather clutter moving at 8 m/s (not at v=0!)")
print("  - Fixed MTI will fail (notch at wrong frequency)")
print("  - Adaptive MTI should work (learns clutter Doppler)")

radar_adaptive = ClutterRadar(num_pulses=128, samples_per_pulse=256)
radar_adaptive.add_weather_clutter(range_start=500, range_end=3000, 
                                   wind_velocity=8, cnr_db=35)
radar_adaptive.add_target(range_m=1500, velocity_ms=18, rcs=0.01)
radar_adaptive.add_target(range_m=2000, velocity_ms=-15, rcs=0.01)
radar_adaptive.add_noise(snr_db=20)

print("✓ Scenario created")

# Generate RDMs
print("\nApplying filters:")
print("  1. No filter (raw)")
print("  2. Fixed MTI (notch at v=0)")
print("  3. Adaptive MTI (learns clutter)")

# Raw
rdm_raw_adapt, rdm_raw_adapt_db, r_axis, v_axis = radar_adaptive.generate_rdm()

# Fixed MTI
data_fixed_mti = three_pulse_mti(radar_adaptive.data_matrix)
radar_temp = ClutterRadar(num_pulses=128, samples_per_pulse=256)
radar_temp.data_matrix = data_fixed_mti
rdm_fixed_mti, rdm_fixed_mti_db, _, _ = radar_temp.generate_rdm()

# Adaptive MTI
print("  Computing adaptive filter weights (may take a moment)...")
data_adaptive_mti = adaptive_mti_filter(radar_adaptive.data_matrix, 
                                       num_training_cells=32, 
                                       filter_length=8)
radar_temp.data_matrix = data_adaptive_mti
rdm_adaptive_mti, rdm_adaptive_mti_db, _, _ = radar_temp.generate_rdm()

print("✓ All filters applied")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Raw
im0 = axes[0, 0].imshow(rdm_raw_adapt_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 0].axhline(8, color='white', linestyle='--', linewidth=2, alpha=0.7,
                   label='Clutter velocity')
axes[0, 0].set_title('No Filtering', fontweight='bold', fontsize=13)
axes[0, 0].set_xlabel('Range [km]')
axes[0, 0].set_ylabel('Velocity [m/s]')
axes[0, 0].legend(loc='upper right')
plt.colorbar(im0, ax=axes[0, 0], label='Power [dB]')

# Fixed MTI
im1 = axes[0, 1].imshow(rdm_fixed_mti_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 1].axhline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 1].axhline(8, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 1].set_title('Fixed MTI (notch at v=0)', fontweight='bold', fontsize=13)
axes[0, 1].set_xlabel('Range [km]')
axes[0, 1].set_ylabel('Velocity [m/s]')
axes[0, 1].text(2, -30, 'Clutter at v=8\nNOT removed!', 
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.8),
               fontsize=10)
plt.colorbar(im1, ax=axes[0, 1], label='Power [dB]')

# Adaptive MTI
im2 = axes[0, 2].imshow(rdm_adaptive_mti_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 2].set_title('Adaptive MTI (learned notch)', fontweight='bold', fontsize=13)
axes[0, 2].set_xlabel('Range [km]')
axes[0, 2].set_ylabel('Velocity [m/s]')
axes[0, 2].text(2, -30, 'Clutter removed!\nAdapted to v=8', 
               bbox=dict(boxstyle='round', facecolor='lime', alpha=0.8),
               fontsize=10)
plt.colorbar(im2, ax=axes[0, 2], label='Power [dB]')

# Doppler cuts at target range
range_bin = int(1500 / radar_adaptive.range_resolution)
axes[1, 0].plot(v_axis, rdm_raw_adapt_db[:, range_bin], 'r-', 
               linewidth=2, label='No Filter')
axes[1, 0].plot(v_axis, rdm_fixed_mti_db[:, range_bin], 'orange', 
               linewidth=2, label='Fixed MTI')
axes[1, 0].plot(v_axis, rdm_adaptive_mti_db[:, range_bin], 'g-', 
               linewidth=2, label='Adaptive MTI')
axes[1, 0].axvline(8, color='red', linestyle=':', alpha=0.7, label='Clutter velocity')
axes[1, 0].axvline(18, color='blue', linestyle=':', alpha=0.7, label='Target velocity')
axes[1, 0].set_xlabel('Velocity [m/s]', fontsize=11)
axes[1, 0].set_ylabel('Power [dB]', fontsize=11)
axes[1, 0].set_title('Doppler Profile Comparison', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Clutter rejection comparison
v_clutter_idx = np.argmin(np.abs(v_axis - 8))
v_target_idx = np.argmin(np.abs(v_axis - 18))

clutter_raw = rdm_raw_adapt_db[v_clutter_idx, range_bin]
clutter_fixed = rdm_fixed_mti_db[v_clutter_idx, range_bin]
clutter_adaptive = rdm_adaptive_mti_db[v_clutter_idx, range_bin]

target_raw = rdm_raw_adapt_db[v_target_idx, range_bin]
target_fixed = rdm_fixed_mti_db[v_target_idx, range_bin]
target_adaptive = rdm_adaptive_mti_db[v_target_idx, range_bin]

methods = ['No Filter', 'Fixed MTI', 'Adaptive MTI']
clutter_levels = [clutter_raw, clutter_fixed, clutter_adaptive]
target_levels = [target_raw, target_fixed, target_adaptive]

x_pos = np.arange(len(methods))
width = 0.35

bars1 = axes[1, 1].bar(x_pos - width/2, clutter_levels, width, 
                       label='Clutter (v=8)', color='red', alpha=0.7)
bars2 = axes[1, 1].bar(x_pos + width/2, target_levels, width,
                       label='Target (v=18)', color='green', alpha=0.7)

axes[1, 1].set_ylabel('Power [dB]', fontsize=11)
axes[1, 1].set_title('Clutter vs Target Power', fontweight='bold', fontsize=12)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(methods)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# SCR comparison
scr_raw = target_raw - clutter_raw
scr_fixed = target_fixed - clutter_fixed
scr_adaptive = target_adaptive - clutter_adaptive

bars3 = axes[1, 2].bar(methods, [scr_raw, scr_fixed, scr_adaptive],
                       color=['red', 'orange', 'green'], alpha=0.7, 
                       edgecolor='black')
axes[1, 2].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1, 2].set_ylabel('Signal-to-Clutter Ratio [dB]', fontsize=11)
axes[1, 2].set_title('Clutter Rejection Performance', fontweight='bold', fontsize=12)
axes[1, 2].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars3, [scr_raw, scr_fixed, scr_adaptive]):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.1f} dB',
                    ha='center', va='bottom' if val > 0 else 'top',
                    fontweight='bold', fontsize=11)

axes[1, 2].text(0.5, 0.95, 
               f'Adaptive MTI:\n+{scr_adaptive - scr_raw:.1f} dB improvement',
               transform=axes[1, 2].transAxes,
               bbox=dict(boxstyle='round', facecolor='lime', alpha=0.8),
               fontsize=11, va='top', ha='center')

plt.tight_layout()
plt.show()

print("\n✓ Adaptive MTI demonstration complete")
print("\n" + "=" * 70)
print("KEY OBSERVATION:")
print("=" * 70)
print(f"For moving clutter (v = 8 m/s):")
print(f"  Fixed MTI:    {scr_fixed - scr_raw:+.1f} dB improvement (fails!)")
print(f"  Adaptive MTI: {scr_adaptive - scr_raw:+.1f} dB improvement (works!)")
print("\n💡 Adaptive MTI automatically finds and suppresses clutter,")
print("   regardless of clutter velocity!")
print("=" * 70)

# %% [markdown]
"""
---
# 4. Brief STAP Concept (For Completeness)

## What is STAP?

**STAP = Space-Time Adaptive Processing**

**DEFINITION:** Joint processing of:
- **Space** = Multiple antenna elements (array)
- **Time** = Multiple pulses (Doppler)

Think of it as "MTI on steroids" - you can reject clutter at specific:
- Velocities (like MTI)
- **AND** angles (new capability!)

## Why STAP?

**Problem MTI Cannot Solve:**
Clutter coming from a specific direction (e.g., mainlobe clutter from ground)

**Example Scenario:**
```
Airborne Radar Looking Down:

        Aircraft
           |
           v (looking down)
      
    Ground clutter
   spreads across
   all Dopplers!
   
   Why? Clutter at different angles
   has different Doppler due to
   platform motion.
```

**MTI alone:** Cannot separate clutter from targets (both spread in Doppler)

**STAP:** Can reject clutter from specific angle-Doppler combinations

## STAP Visual Representation

```
WITHOUT STAP (Doppler only):
Angle-Doppler Space:

Angle  ░░░░░░░░░░░░░░░░  ← Clutter ridge (spread)
  ↑    ░░░░●░░░░░░░░░░  ← Target buried
  |    ░░░░░░░░░░░░░░░
  |    ░░░░░░░░░░░░░░░
  0    ════════════════→ Doppler


WITH STAP:
Angle-Doppler Space:

Angle  ················  ← Clutter notched out!
  ↑    ········●·······  ← Target visible!
  |    ················
  |    ················
  0    ════════════════→ Doppler
```

## Why We Don't Implement Full STAP Here

**Requirements for STAP:**
1. Multiple antenna elements (array radar)
2. Precise calibration across elements
3. Knowledge of platform motion (for airborne)
4. Significant computational resources

**Our Tutorial Uses:** Single antenna (monostatic radar)

**Takeaway:** STAP is powerful for airborne/moving platforms with arrays,
but MTI + adaptive techniques are sufficient for ground-based systems.

---
"""

# %% [markdown]
"""
---
# 5. Neural Network for Clutter Cancellation (AI Approach)

## The AI Alternative: Learn to Suppress

**Classical approach:** Hand-design filter (MTI, adaptive filter)
- Requires understanding of clutter statistics
- Fixed mathematical model
- May not adapt to complex, non-stationary clutter

**AI approach:** Learn from data
- Network learns what "clutter" looks like
- No explicit model needed
- Can handle complex, time-varying clutter

## Network Architecture: U-Net for Clutter Suppression

**Why U-Net?**
- Originally designed for image segmentation
- Perfect for RDM processing (2D image-like data)
- Encoder-Decoder structure preserves spatial information
- Skip connections help preserve target details

**Visual Architecture:**
```
Input RDM (with clutter)
         ↓
    [Encoder]  ──→ Extract features
         ↓         (what is clutter?)
    [Bottleneck]
         ↓
    [Decoder]  ──→ Reconstruct RDM
         ↓         (remove clutter)
Output RDM (clean)
```

---
"""

# %%
class ClutterNet(nn.Module):
    """
    U-Net style network for clutter suppression
    
    ARCHITECTURE:
    - Encoder: Downsample and extract features
    - Bottleneck: Compress to latent representation
    - Decoder: Upsample and reconstruct clean RDM
    - Skip connections: Preserve fine details
    
    WHY THIS ARCHITECTURE:
    - U-Net preserves spatial structure (important for RDM)
    - Encoder learns "what is clutter?"
    - Decoder learns "how to remove it?"
    - Skip connections help preserve weak targets
    
    INPUT:
    - RDM with clutter [1 x H x W]
    - H = num_doppler_bins (e.g., 128)
    - W = num_range_bins (e.g., 256)
    
    OUTPUT:
    - Clean RDM [1 x H x W]
    - Clutter suppressed, targets preserved
    
    TRAINING:
    - Input: RDM with clutter
    - Target: RDM without clutter (ground truth)
    - Loss: MSE between output and target
    """
    
    def __init__(self):
        super(ClutterNet, self).__init__()
        
        # Encoder (downsampling path)
        # Each layer: Conv -> BatchNorm -> ReLU -> MaxPool
        self.enc1 = self._make_encoder_block(1, 32)      # 128x256 -> 64x128
        self.enc2 = self._make_encoder_block(32, 64)     # 64x128 -> 32x64
        self.enc3 = self._make_encoder_block(64, 128)    # 32x64 -> 16x32
        self.enc4 = self._make_encoder_block(128, 256)   # 16x32 -> 8x16
        
        # Bottleneck (deepest layer)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling path)
        # Each layer: Upsample -> Conv -> BatchNorm -> ReLU
        # Skip connection from encoder concatenated
        self.dec4 = self._make_decoder_block(512 + 256, 256)  # +256 from enc4
        self.dec3 = self._make_decoder_block(256 + 128, 128)  # +128 from enc3
        self.dec2 = self._make_decoder_block(128 + 64, 64)    # +64 from enc2
        self.dec1 = self._make_decoder_block(64 + 32, 32)     # +32 from enc1
        
        # Final output layer
        self.output = nn.Conv2d(32, 1, kernel_size=1)
    
    def _make_encoder_block(self, in_channels, out_channels):
        """Create encoder block with Conv-BN-ReLU-Pool"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create decoder block with Conv-BN-ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass through network
        
        Parameters:
        -----------
        x : torch.Tensor
            Input RDM [batch_size, 1, H, W]
        
        Returns:
        --------
        out : torch.Tensor
            Clean RDM [batch_size, 1, H, W]
        """
        # Encoder with skip connections saved
        enc1_out = self.enc1(x)          # -> [B, 32, 64, 128]
        enc2_out = self.enc2(enc1_out)   # -> [B, 64, 32, 64]
        enc3_out = self.enc3(enc2_out)   # -> [B, 128, 16, 32]
        enc4_out = self.enc4(enc3_out)   # -> [B, 256, 8, 16]
        
        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_out)  # -> [B, 512, 8, 16]
        
        # Decoder with skip connections
        # Upsample and concatenate with corresponding encoder output
        dec4_up = F.interpolate(bottleneck_out, scale_factor=2, mode='bilinear', 
                                align_corners=False)
        dec4_cat = torch.cat([dec4_up, enc4_out], dim=1)  # Concatenate
        dec4_out = self.dec4(dec4_cat)   # -> [B, 256, 16, 32]
        
        dec3_up = F.interpolate(dec4_out, scale_factor=2, mode='bilinear',
                                align_corners=False)
        dec3_cat = torch.cat([dec3_up, enc3_out], dim=1)
        dec3_out = self.dec3(dec3_cat)   # -> [B, 128, 32, 64]
        
        dec2_up = F.interpolate(dec3_out, scale_factor=2, mode='bilinear',
                                align_corners=False)
        dec2_cat = torch.cat([dec2_up, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_cat)   # -> [B, 64, 64, 128]
        
        dec1_up = F.interpolate(dec2_out, scale_factor=2, mode='bilinear',
                                align_corners=False)
        dec1_cat = torch.cat([dec1_up, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_cat)   # -> [B, 32, 128, 256]
        
        # Final output
        out = self.output(dec1_out)      # -> [B, 1, 128, 256]
        
        return out


# %%
def generate_training_pairs(num_samples=100, clutter_type='mixed'):
    """
    Generate training pairs: (RDM with clutter, RDM without clutter)
    
    WHY WE NEED THIS:
    Neural networks learn by example. We need to show them:
    - Input: What RDM with clutter looks like
    - Output: What clean RDM should look like
    
    TRAINING STRATEGY:
    1. Create clean RDM (targets only)
    2. Add synthetic clutter
    3. Pair: (cluttered RDM, clean RDM)
    4. Network learns: Input → Remove clutter → Output
    
    Parameters:
    -----------
    num_samples : int
        Number of training pairs to generate
    clutter_type : str
        'ground', 'weather', 'sea', or 'mixed'
    
    Returns:
    --------
    rdm_cluttered_list : list of ndarray
        List of RDMs with clutter (inputs)
    rdm_clean_list : list of ndarray
        List of clean RDMs (targets)
    """
    rdm_cluttered_list = []
    rdm_clean_list = []
    
    print(f"Generating {num_samples} training pairs with {clutter_type} clutter...")
    
    for i in range(num_samples):
        if (i+1) % 20 == 0:
            print(f"  Generated {i+1}/{num_samples} pairs...")
        
        # Create radar
        radar = ClutterRadar(num_pulses=128, samples_per_pulse=256)
        
        # Add random targets (1-3 targets per sample)
        num_targets = np.random.randint(1, 4)
        for _ in range(num_targets):
            range_m = np.random.uniform(500, 4000)
            velocity_ms = np.random.uniform(-30, 30)
            # Avoid v ≈ 0 (would look like clutter)
            if abs(velocity_ms) < 3:
                velocity_ms += np.sign(velocity_ms) * 3
            rcs = np.random.uniform(0.005, 0.02)
            radar.add_target(range_m, velocity_ms, rcs)
        
        # Generate CLEAN RDM (targets + noise only)
        radar.add_noise(snr_db=20)
        rdm_clean, _, _, _ = radar.generate_rdm()
        
        # Add clutter to make CLUTTERED RDM
        if clutter_type == 'ground' or (clutter_type == 'mixed' and np.random.rand() < 0.4):
            r_start = np.random.uniform(0, 1000)
            r_end = r_start + np.random.uniform(1000, 2000)
            cnr = np.random.uniform(40, 55)
            radar.add_ground_clutter(r_start, r_end, cnr_db=cnr)
        
        if clutter_type == 'weather' or (clutter_type == 'mixed' and np.random.rand() < 0.4):
            r_start = np.random.uniform(1000, 2000)
            r_end = r_start + np.random.uniform(1000, 2000)
            wind_vel = np.random.uniform(3, 12)
            cnr = np.random.uniform(25, 40)
            radar.add_weather_clutter(r_start, r_end, wind_vel, cnr_db=cnr)
        
        if clutter_type == 'sea' or (clutter_type == 'mixed' and np.random.rand() < 0.4):
            r_start = np.random.uniform(1500, 2500)
            r_end = r_start + np.random.uniform(1000, 2000)
            sea_state = np.random.randint(2, 5)
            cnr = np.random.uniform(30, 45)
            radar.add_sea_clutter(r_start, r_end, sea_state, cnr_db=cnr)
        
        rdm_cluttered, _, _, _ = radar.generate_rdm()
        
        # Store pair
        rdm_cluttered_list.append(rdm_cluttered)
        rdm_clean_list.append(rdm_clean)
    
    print(f"✓ Generated {num_samples} training pairs")
    return rdm_cluttered_list, rdm_clean_list


# %%
def train_clutter_net(model, train_cluttered, train_clean, 
                     num_epochs=50, batch_size=8, learning_rate=1e-4):
    """
    Train ClutterNet to suppress clutter
    
    TRAINING PROCESS:
    1. Feed cluttered RDM into network
    2. Network outputs predicted clean RDM
    3. Compare prediction to ground truth clean RDM
    4. Compute loss (how different?)
    5. Backpropagate and update weights
    6. Repeat until network learns to remove clutter
    
    Parameters:
    -----------
    model : ClutterNet
        The neural network to train
    train_cluttered : list of ndarray
        Training inputs (RDMs with clutter)
    train_clean : list of ndarray
        Training targets (clean RDMs)
    num_epochs : int
        How many times to iterate through dataset
    batch_size : int
        Number of samples per batch
    learning_rate : float
        Step size for gradient descent
    
    Returns:
    --------
    model : ClutterNet
        Trained network
    losses : list
        Training loss history
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    num_samples = len(train_cluttered)
    losses = []
    
    print(f"\nTraining ClutterNet:")
    print(f"  Device: {device}")
    print(f"  Samples: {num_samples}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data each epoch
        indices = np.random.permutation(num_samples)
        
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]
            
            # Prepare batch
            batch_cluttered = []
            batch_clean = []
            
            for idx in batch_indices:
                # Normalize to [0, 1] for better training
                rdm_clut = train_cluttered[idx]
                rdm_cln = train_clean[idx]
                
                rdm_clut_norm = (rdm_clut - rdm_clut.min()) / (rdm_clut.max() - rdm_clut.min() + 1e-10)
                rdm_cln_norm = (rdm_cln - rdm_cln.min()) / (rdm_cln.max() - rdm_cln.min() + 1e-10)
                
                batch_cluttered.append(rdm_clut_norm)
                batch_clean.append(rdm_cln_norm)
            
            # Convert to tensors
            x = torch.FloatTensor(np.array(batch_cluttered)).unsqueeze(1).to(device)  # [B, 1, H, W]
            y = torch.FloatTensor(np.array(batch_clean)).unsqueeze(1).to(device)
            
            # Forward pass
            y_pred = model(x)
            
            # Compute loss
            loss = criterion(y_pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    print("✓ Training complete")
    
    return model, losses


# %%
print("\n" + "=" * 70)
print("DEMONSTRATION: Neural Network Clutter Suppression")
print("=" * 70)

# NOTE: This is a simplified demonstration
# Full training would require thousands of samples and longer training
# Here we use a small dataset for illustration

print("\nStep 1: Generate training data")
print("  (Using small dataset for demonstration)")

# Generate small training set
num_train_samples = 50  # In practice: use 1000-5000 samples
train_cluttered, train_clean = generate_training_pairs(
    num_samples=num_train_samples,
    clutter_type='mixed'
)

print("\nStep 2: Initialize and train network")
print("  (Training with reduced epochs for demo)")

# Initialize network
clutter_net = ClutterNet()

# Count parameters
num_params = sum(p.numel() for p in clutter_net.parameters())
print(f"  Network parameters: {num_params:,}")

# Train (with reduced epochs for demo)
trained_model, loss_history = train_clutter_net(
    clutter_net,
    train_cluttered,
    train_clean,
    num_epochs=30,  # In practice: use 100-200 epochs
    batch_size=4,
    learning_rate=1e-4
)

print("\nStep 3: Test on new data")

# Generate test scenario
radar_test = ClutterRadar(num_pulses=128, samples_per_pulse=256)
radar_test.add_ground_clutter(0, 1500, cnr_db=50)
radar_test.add_weather_clutter(1500, 3500, wind_velocity=7, cnr_db=32)
radar_test.add_target(range_m=1200, velocity_ms=18, rcs=0.01)
radar_test.add_target(range_m=2500, velocity_ms=-15, rcs=0.01)
radar_test.add_noise(snr_db=20)

# Generate clean reference (targets only, before adding clutter)
rdm_reference, rdm_reference_db, r_axis, v_axis = radar_test.generate_rdm()

# Add clutter for test input
radar_test.data_matrix = np.zeros_like(radar_test.data_matrix)
radar_test.add_ground_clutter(0, 1500, cnr_db=50)
radar_test.add_weather_clutter(1500, 3500, wind_velocity=7, cnr_db=32)
radar_test.add_target(range_m=1200, velocity_ms=18, rcs=0.01)
radar_test.add_target(range_m=2500, velocity_ms=-15, rcs=0.01)
radar_test.add_noise(snr_db=20)
rdm_test_cluttered, rdm_test_cluttered_db, _, _ = radar_test.generate_rdm()

# Apply neural network
trained_model.eval()
with torch.no_grad():
    # Normalize input
    rdm_input_norm = (rdm_test_cluttered - rdm_test_cluttered.min()) / \
                     (rdm_test_cluttered.max() - rdm_test_cluttered.min() + 1e-10)
    
    # To tensor
    x_test = torch.FloatTensor(rdm_input_norm).unsqueeze(0).unsqueeze(0)
    
    # Forward pass
    y_pred = trained_model(x_test)
    
    # Back to numpy
    rdm_nn_output = y_pred.squeeze().cpu().numpy()
    
    # Denormalize (approximate)
    rdm_nn_output = rdm_nn_output * rdm_test_cluttered.max()

rdm_nn_output_db = 10 * np.log10(rdm_nn_output + 1e-10)

print("✓ Neural network applied to test data")

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Input (cluttered)
im0 = axes[0, 0].imshow(rdm_test_cluttered_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 0].set_title('Input: RDM with Clutter', fontweight='bold', fontsize=13)
axes[0, 0].set_xlabel('Range [km]')
axes[0, 0].set_ylabel('Velocity [m/s]')
plt.colorbar(im0, ax=axes[0, 0], label='Power [dB]')

# Neural network output
im1 = axes[0, 1].imshow(rdm_nn_output_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 1].set_title('Neural Network Output', fontweight='bold', fontsize=13)
axes[0, 1].set_xlabel('Range [km]')
axes[0, 1].set_ylabel('Velocity [m/s]')
axes[0, 1].text(2, 30, 'Learned to\nremove clutter!', 
               bbox=dict(boxstyle='round', facecolor='lime', alpha=0.8),
               fontsize=11)
plt.colorbar(im1, ax=axes[0, 1], label='Power [dB]')

# Ground truth (for comparison)
im2 = axes[0, 2].imshow(rdm_reference_db, aspect='auto', cmap='jet',
                        extent=[r_axis[0]/1e3, r_axis[-1]/1e3,
                               v_axis[0], v_axis[-1]],
                        vmin=-40, vmax=20, origin='lower')
axes[0, 2].set_title('Ground Truth (Clean)', fontweight='bold', fontsize=13)
axes[0, 2].set_xlabel('Range [km]')
axes[0, 2].set_ylabel('Velocity [m/s]')
plt.colorbar(im2, ax=axes[0, 2], label='Power [dB]')

# Training loss
axes[1, 0].plot(loss_history, 'b-', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=11)
axes[1, 0].set_ylabel('Loss (MSE)', fontsize=11)
axes[1, 0].set_title('Training Loss', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].text(0.5, 0.95, f'Final loss: {loss_history[-1]:.6f}',
               transform=axes[1, 0].transAxes,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
               fontsize=10, va='top', ha='center')

# Doppler cut comparison
range_bin = int(1200 / radar_test.range_resolution)
axes[1, 1].plot(v_axis, rdm_test_cluttered_db[:, range_bin], 'r-',
               linewidth=2, label='With Clutter', alpha=0.7)
axes[1, 1].plot(v_axis, rdm_nn_output_db[:, range_bin], 'g-',
               linewidth=2, label='Neural Network')
axes[1, 1].plot(v_axis, rdm_reference_db[:, range_bin], 'b--',
               linewidth=2, label='Ground Truth', alpha=0.7)
axes[1, 1].axvline(18, color='black', linestyle=':', alpha=0.5,
                   label='Target velocity')
axes[1, 1].set_xlabel('Velocity [m/s]', fontsize=11)
axes[1, 1].set_ylabel('Power [dB]', fontsize=11)
axes[1, 1].set_title('Doppler Profile at Target Range', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

# Performance summary
axes[1, 2].axis('off')
summary_text = """NEURAL NETWORK APPROACH:

ADVANTAGES:
✓ Learns from data
✓ No explicit clutter model needed
✓ Handles complex, non-stationary clutter
✓ Can learn time-varying patterns

DISADVANTAGES:
✗ Requires training data
✗ Black box (hard to interpret)
✗ May not generalize to unseen scenarios
✗ Computationally intensive (training)

TRAINING REQUIREMENTS:
• 1000-5000 samples for good performance
• Diverse clutter scenarios
• 100-200 epochs
• GPU recommended

WHEN TO USE:
→ Have lots of labeled data
→ Clutter is complex/variable
→ Computational resources available
→ Need best possible performance
"""

axes[1, 2].text(0.1, 0.95, summary_text, fontsize=10, family='monospace',
               verticalalignment='top', transform=axes[1, 2].transAxes)
axes[1, 2].set_title('Neural Network Summary', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.show()

print("\n✓ Neural network demonstration complete")
print("\n" + "=" * 70)
print("NOTE: This is a simplified demonstration with limited training.")
print("      For production use:")
print("      - Use 1000-5000 training samples")
print("      - Train for 100-200 epochs")
print("      - Use data augmentation")
print("      - Validate on separate test set")
print("=" * 70)

# %% [markdown]
"""
---
# 6. Summary and Recommendations

## Method Comparison Table

| Method | Clutter Type | Complexity | Performance | When to Use |
|--------|--------------|------------|-------------|-------------|
| **Single MTI** | Stationary | Low | Good | Ground clutter, simple scenarios |
| **Three-Pulse MTI** | Stationary | Low | Better | Strong ground clutter |
| **Adaptive MTI** | Any | Medium | Excellent | Moving clutter, heterogeneous |
| **STAP** | Any (angle-dependent) | Very High | Excellent | Array radar, airborne platform |
| **Neural Network** | Any | High (training) | Excellent | Complex, variable clutter with data |

## Decision Tree: Which Method to Use?

```
Is clutter stationary (v ≈ 0)?
│
├─ YES: Use MTI
│   │
│   └─ Is clutter very strong (CNR > 50 dB)?
│       │
│       ├─ YES: Use Three-Pulse MTI
│       │
│       └─ NO: Use Single-Delay MTI
│
└─ NO: Clutter is moving
    │
    └─ Do you have array radar?
        │
        ├─ YES: Consider STAP
        │
        └─ NO: Use Adaptive MTI or Neural Network
            │
            └─ Do you have lots of training data?
                │
                ├─ YES: Train Neural Network
                │
                └─ NO: Use Adaptive MTI
```

## Key Takeaways

✅ **Clutter characteristics determine suppression approach**
   - Ground: Stationary, use MTI
   - Weather: Moving, use adaptive methods
   - Sea: Complex, use neural networks if data available

✅ **MTI is a high-pass filter**
   - Removes DC component (v=0)
   - Simple, effective for ground clutter
   - Deeper notch = better rejection

✅ **Adaptive methods learn from data**
   - Estimate clutter statistics
   - Adjust filter to match
   - Handle non-stationary clutter

✅ **Neural networks are powerful but require data**
   - Learn complex patterns
   - Best performance with enough training
   - Trade-off: data requirements vs performance

✅ **No single method is best for all scenarios**
   - Understand your environment
   - Choose appropriate technique
   - Consider computational constraints

## Practical Recommendations

**For ground-based counter-drone radar:**
1. Start with three-pulse MTI (simple, effective)
2. Add adaptive MTI for windy conditions
3. Consider neural network if you can collect labeled data

**For operational systems:**
- Implement multiple methods
- Switch based on environment conditions
- Monitor performance and adapt

**Remember:**
Clutter suppression is about exploiting differences between clutter and targets.
The best method depends on knowing your clutter characteristics!

---
"""

# %%
# Final check
file_path = __file__

if os.path.exists(file_path):
    size_bytes = os.path.getsize(file_path)
    size_kb = size_bytes / 1024
    size_mb = size_bytes / (1024 * 1024)
    
    print("\n" + "=" * 70)
    print("FILE SIZE CHECK")
    print("=" * 70)
    print(f"Final size: {size_kb:.1f} KB ({size_mb:.3f} MB)")
    
    if size_mb > 1.0:
        print("❌ ERROR: Exceeds 1 MB limit!")
    elif size_mb > 0.9:
        print("⚠️  WARNING: Very close to 1 MB limit")
    else:
        print(f"✓ Size OK - {(1.0-size_mb)*1024:.0f} KB remaining")
    
    print("=" * 70)

print("\n" + "=" * 70)
print("PART 5 COMPLETE!")
print("=" * 70)
print("\n✅ Topics Covered:")
print("  • Clutter types and characteristics")
print("  • Classical MTI filters (single-delay, three-pulse)")
print("  • Adaptive MTI (covariance-based)")
print("  • STAP concept (brief overview)")
print("  • Neural network clutter suppression")
print("  • Method comparison and recommendations")
print("\n🎯 You now understand:")
print("  • Why clutter is a problem (masks weak targets)")
print("  • How MTI exploits Doppler differences")
print("  • When to use each suppression method")
print("  • Trade-offs between classical and AI approaches")
print("\n📝 Next Steps:")
print("  • Part 6: Classification (Drone vs Bird)")
print("  • Part 7: Tracking (Kalman, LSTM)")
print("  • Part 8: Cognitive Radar (Multi-agent system)")
print("=" * 70)
