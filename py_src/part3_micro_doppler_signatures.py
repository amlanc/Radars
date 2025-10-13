# %% [markdown]
"""
RADAR FUNDAMENTALS & SIMULATOR - Part 3
========================================
Micro-Doppler Signatures & Time-Frequency Analysis

Target: Understand and extract micro-Doppler features for drone detection
Timeline: Weekend 2 (Oct 12-13)

Prerequisites: Parts 1 & 2 (RDM generation understanding)
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
"""
---
# Part 3: Micro-Doppler Signatures

## Recap from Part 2

You now understand:
- ✅ Range-Doppler Map (RDM): 2D view of targets (range vs velocity)
- ✅ Doppler shift from moving targets
- ✅ MTI filtering to remove stationary clutter

## The Drone Detection Problem

**Challenge:** In an RDM, a drone and a bird at the same range/velocity look identical!
- Both show up as a single bright spot
- Same Doppler frequency (body motion)
- Similar RCS (radar cross section)

**How do we tell them apart?**

**Answer:** **Micro-Doppler signatures** - the unique "fingerprint" from rotating parts!

---
# 1. What is Micro-Doppler?

## Think Back to Basics

Imagine you're watching a ceiling fan from across the room:
- The **fan body** is stationary → no Doppler
- But the **blades** are rotating → they move toward AND away from you!

**At each instant:**
- Blade tip moving toward you → Positive Doppler (+)
- Blade tip moving away → Negative Doppler (−)
- Blade at 90° to you → No radial velocity, zero Doppler

**As the blade rotates:**
- Doppler frequency **oscillates** between + and − at the rotation rate
- This creates a **time-varying Doppler signature** on top of the body's constant Doppler

## Micro-Doppler in Radar

**Main Doppler** = Body motion (drone flying at 15 m/s)
**Micro-Doppler** = Rotating parts (propellers at 3000 RPM)

**Key signatures:**
- **Drones:** Periodic blade flashes (4 blades → 4 flashes per rotation)
- **Birds:** Irregular wing beats (2 wings, non-uniform motion)
- **Helicopters:** Strong rotor modulation (large blades)

**This is THE signature that makes drone detection possible!**

## Why RDM Can't Show This

Remember:
- RDM shows **one Doppler value per target** (averaged over CPI)
- CPI = 128 pulses ≈ 12.8 ms (for PRF = 10 kHz)
- Blade rotation period ≈ 20 ms (for 3000 RPM)

**Problem:** The fast blade oscillations are **hidden** in the slow pulse-to-pulse measurements!

**Solution:** Look at **time-frequency** evolution using **spectrograms**
"""

# %% [markdown]
"""
---
# 2. Physics of Rotor Blade Modulation

## Blade Velocity Calculation

Consider a single rotor blade:
- Length: `r_blade` (e.g., 15 cm)
- Angular velocity: `ω` (e.g., 3000 RPM = 314 rad/s)
- Blade orientation: `θ(t) = ωt`

**Blade tip velocity** (relative to drone body):
```
v_blade(t) = r_blade × ω × sin(θ(t))
           = r_blade × ω × sin(ωt)
```

**Radial component** (toward/away from radar):
```
v_radial(t) = v_blade(t) × cos(aspect_angle)
```

## Micro-Doppler Frequency

From Doppler formula: `f_d = 2v/λ`

**Micro-Doppler from one blade:**
```
f_micro(t) = (2/λ) × r_blade × ω × sin(ωt)
           = f_max × sin(ωt)

Where: f_max = (2/λ) × r_blade × ω
```

**For multiple blades** (e.g., quadcopter with 4 blades):
- Each blade is 90° offset
- Creates 4 peaks per rotation
- **Blade flash frequency** = Number of blades × rotation rate

## Example Calculation

**Typical quadcopter:**
```
r_blade = 0.15 m
RPM = 3000 → ω = 314 rad/s
λ = 0.03 m (10 GHz X-band)

f_max = 2 × 0.15 × 314 / 0.03
      = 3,140 Hz

Blade flash rate = 4 blades × 50 Hz (rotation)
                 = 200 Hz
```

**This 200 Hz modulation is what we'll extract from spectrograms!**
"""

# %%
# Visualization: Single blade micro-Doppler
def visualize_blade_modulation():
    """
    Show how a single rotating blade creates sinusoidal Doppler modulation
    """
    # Parameters
    r_blade = 0.15  # meters
    rpm = 3000
    omega = rpm * 2 * np.pi / 60  # rad/s
    wavelength = 0.03  # 10 GHz
    f_c = 10e9  # carrier frequency
    
    # Time array (1 rotation)
    t = np.linspace(0, 2*np.pi/omega, 1000)
    
    # Blade angle
    theta = omega * t
    
    # Radial velocity of blade tip
    v_radial = r_blade * omega * np.sin(theta)
    
    # Micro-Doppler frequency
    f_micro = 2 * v_radial / wavelength
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Blade position and velocity
    ax = axes[0]
    ax.plot(t * 1000, v_radial, 'b-', linewidth=2, label='Radial velocity')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(t * 1000, 0, v_radial, alpha=0.3)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Radial Velocity (m/s)')
    ax.set_title('Blade Tip Radial Velocity (One Rotation)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations
    max_idx = np.argmax(v_radial)
    ax.annotate('Blade moving\ntoward radar', 
                xy=(t[max_idx]*1000, v_radial[max_idx]),
                xytext=(t[max_idx]*1000 + 5, v_radial[max_idx] + 20),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold')
    
    # Plot 2: Micro-Doppler frequency
    ax = axes[1]
    ax.plot(t * 1000, f_micro, 'r-', linewidth=2, label='Micro-Doppler frequency')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(t * 1000, 0, f_micro, alpha=0.3, color='red')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Micro-Doppler Frequency Modulation', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add parameters text
    param_text = f'Blade: {r_blade*100:.0f} cm | RPM: {rpm} | λ: {wavelength*100:.1f} cm'
    ax.text(0.02, 0.95, param_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✓ Peak micro-Doppler: ±{np.max(f_micro):.1f} Hz")
    print(f"✓ Modulation frequency: {rpm/60:.1f} Hz (rotation rate)")

# Run visualization
visualize_blade_modulation()

# %% [markdown]
"""
---
# 3. Generating Micro-Doppler Signatures

## The MicroDopplerRadar Class

We'll build a radar simulator that can:
1. Simulate a moving target with rotating blades
2. Generate time-series radar returns
3. Analyze micro-Doppler using spectrograms

**Key difference from Part 2:**
- Part 2: Multiple pulses → 2D FFT → RDM (range-velocity)
- Part 3: Continuous time series → STFT → Spectrogram (time-frequency)
"""

# %%
class MicroDopplerRadar:
    """
    Radar simulator for micro-Doppler analysis
    
    Simulates continuous radar returns from targets with rotating parts
    Focus: Time-frequency analysis, not range-velocity maps
    """
    
    def __init__(self, f_c=10e9, prf=10e3, observation_time=1.0,
                 sampling_rate=None):
        """
        Initialize micro-Doppler radar simulator
        
        Parameters:
        -----------
        f_c : float
            Carrier frequency in Hz (default: 10 GHz X-band)
        prf : float
            Pulse repetition frequency in Hz (default: 10 kHz)
        observation_time : float
            Total observation time in seconds (default: 1.0 s)
        sampling_rate : float
            Sampling rate for time series (default: 2*prf)
        """
        self.f_c = f_c
        self.wavelength = 3e8 / f_c
        self.prf = prf
        self.pri = 1.0 / prf
        self.observation_time = observation_time
        
        # Sampling rate for continuous simulation
        if sampling_rate is None:
            self.sampling_rate = 2 * prf  # Nyquist
        else:
            self.sampling_rate = sampling_rate
        
        # Time vector
        self.num_samples = int(observation_time * self.sampling_rate)
        self.time_vector = np.arange(self.num_samples) / self.sampling_rate
        
        # Signal storage
        self.signal = np.zeros(self.num_samples, dtype=complex)
        
        # Target list
        self.targets = []
        
        print(f"MicroDopplerRadar initialized:")
        print(f"  Carrier freq: {f_c/1e9:.1f} GHz")
        print(f"  Wavelength: {self.wavelength*100:.2f} cm")
        print(f"  PRF: {prf/1e3:.1f} kHz")
        print(f"  Observation time: {observation_time:.2f} s")
        print(f"  Samples: {self.num_samples:,}")
        print(f"  Sampling rate: {self.sampling_rate/1e3:.1f} kHz")
    
    def add_target_with_rotors(self, range_m, velocity_ms, rcs=1.0,
                               num_rotors=4, rotor_radius=0.15, rpm=3000,
                               blade_rcs_fraction=0.1):
        """
        Add a target with rotating blades (e.g., drone)
        
        Parameters:
        -----------
        range_m : float
            Target range in meters
        velocity_ms : float
            Target radial velocity in m/s (positive = approaching)
        rcs : float
            Radar cross section of main body in m²
        num_rotors : int
            Number of rotor blades
        rotor_radius : float
            Rotor blade radius in meters
        rpm : float
            Rotor rotation speed in RPM
        blade_rcs_fraction : float
            RCS of blade relative to body (typically 0.05 - 0.2)
        """
        target = {
            'range': range_m,
            'velocity': velocity_ms,
            'rcs': rcs,
            'num_rotors': num_rotors,
            'rotor_radius': rotor_radius,
            'rpm': rpm,
            'omega': rpm * 2 * np.pi / 60,  # rad/s
            'blade_rcs': rcs * blade_rcs_fraction,
            'type': 'rotor'
        }
        self.targets.append(target)
        
        print(f"\n✓ Added rotor target:")
        print(f"  Range: {range_m} m | Velocity: {velocity_ms} m/s")
        print(f"  Rotors: {num_rotors} | RPM: {rpm} | Radius: {rotor_radius*100:.1f} cm")
    
    def add_target_with_wings(self, range_m, velocity_ms, rcs=1.0,
                             wing_beat_freq=8, wing_length=0.25,
                             irregularity=0.3):
        """
        Add a target with flapping wings (e.g., bird)
        
        Parameters:
        -----------
        range_m : float
            Target range in meters
        velocity_ms : float
            Target radial velocity in m/s
        rcs : float
            Radar cross section in m²
        wing_beat_freq : float
            Wing beat frequency in Hz (birds: 5-15 Hz)
        wing_length : float
            Wing length in meters
        irregularity : float
            Wing beat irregularity factor (0=regular, 1=very irregular)
        """
        target = {
            'range': range_m,
            'velocity': velocity_ms,
            'rcs': rcs,
            'wing_beat_freq': wing_beat_freq,
            'wing_length': wing_length,
            'irregularity': irregularity,
            'type': 'wing'
        }
        self.targets.append(target)
        
        print(f"\n✓ Added wing target (bird):")
        print(f"  Range: {range_m} m | Velocity: {velocity_ms} m/s")
        print(f"  Wing beat: {wing_beat_freq} Hz | Length: {wing_length*100:.1f} cm")
    
    def generate_signal(self):
        """
        Generate radar return signal with micro-Doppler modulation
        """
        # Reset signal
        self.signal = np.zeros(self.num_samples, dtype=complex)
        
        for target in self.targets:
            if target['type'] == 'rotor':
                signal_component = self._generate_rotor_signal(target)
            elif target['type'] == 'wing':
                signal_component = self._generate_wing_signal(target)
            else:
                signal_component = self._generate_simple_signal(target)
            
            self.signal += signal_component
        
        return self.signal
    
    def _generate_rotor_signal(self, target):
        """
        Generate signal from target with rotating blades
        """
        # Body Doppler (constant velocity)
        body_doppler = 2 * target['velocity'] / self.wavelength
        body_phase = 2 * np.pi * body_doppler * self.time_vector
        
        # Body return (main signal)
        body_signal = np.sqrt(target['rcs']) * np.exp(1j * body_phase)
        
        # Blade micro-Doppler (time-varying)
        omega = target['omega']
        r_blade = target['rotor_radius']
        num_blades = target['num_rotors']
        
        blade_signal = np.zeros(self.num_samples, dtype=complex)
        
        # Each blade contributes
        for blade_idx in range(num_blades):
            # Phase offset for this blade
            blade_offset = blade_idx * (2 * np.pi / num_blades)
            
            # Blade angle over time
            theta = omega * self.time_vector + blade_offset
            
            # Radial velocity component
            v_blade = r_blade * omega * np.sin(theta)
            
            # Micro-Doppler frequency
            f_micro = 2 * v_blade / self.wavelength
            
            # Phase accumulation
            phase_micro = 2 * np.pi * np.cumsum(f_micro) / self.sampling_rate
            
            # Blade return (modulated)
            blade_signal += np.sqrt(target['blade_rcs']) * np.exp(1j * (body_phase + phase_micro))
        
        # Total signal
        return body_signal + blade_signal
    
    def _generate_wing_signal(self, target):
        """
        Generate signal from target with flapping wings (bird)
        """
        # Body Doppler
        body_doppler = 2 * target['velocity'] / self.wavelength
        body_phase = 2 * np.pi * body_doppler * self.time_vector
        
        # Body return
        body_signal = np.sqrt(target['rcs']) * np.exp(1j * body_phase)
        
        # Wing modulation (irregular)
        wing_freq = target['wing_beat_freq']
        wing_length = target['wing_length']
        irregularity = target['irregularity']
        
        # Add irregularity to wing beat
        phase_noise = irregularity * np.cumsum(np.random.randn(self.num_samples)) / self.sampling_rate
        wing_angle = 2 * np.pi * wing_freq * self.time_vector + phase_noise
        
        # Wing velocity (non-sinusoidal, more realistic)
        v_wing = wing_length * wing_freq * np.sin(wing_angle) * (1 + 0.5 * np.cos(2 * wing_angle))
        
        # Micro-Doppler from wings
        f_micro = 2 * v_wing / self.wavelength
        phase_micro = 2 * np.pi * np.cumsum(f_micro) / self.sampling_rate
        
        # Wing contribution (smaller than body)
        wing_signal = 0.3 * np.sqrt(target['rcs']) * np.exp(1j * (body_phase + phase_micro))
        
        return body_signal + wing_signal
    
    def _generate_simple_signal(self, target):
        """
        Generate simple signal (no micro-Doppler)
        """
        doppler = 2 * target['velocity'] / self.wavelength
        phase = 2 * np.pi * doppler * self.time_vector
        return np.sqrt(target['rcs']) * np.exp(1j * phase)
    
    def add_noise(self, snr_db=20):
        """
        Add complex white Gaussian noise
        """
        signal_power = np.mean(np.abs(self.signal)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power/2) * (np.random.randn(self.num_samples) + 
                                          1j*np.random.randn(self.num_samples))
        self.signal += noise
        
        print(f"\n✓ Added noise: SNR = {snr_db} dB")
    
    def generate_spectrogram(self, nperseg=256, noverlap=None, window='hann'):
        """
        Generate spectrogram using Short-Time Fourier Transform (STFT)
        
        Parameters:
        -----------
        nperseg : int
            Length of each segment (affects frequency resolution)
        noverlap : int
            Number of overlapping samples (affects time resolution)
        window : str
            Window function ('hann', 'hamming', 'blackman')
        
        Returns:
        --------
        spectrogram : 2D array
            Time-frequency magnitude spectrogram
        time_axis : array
            Time axis in seconds
        freq_axis : array
            Frequency axis in Hz (Doppler frequencies)
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Compute STFT
        f, t, Zxx = signal.stft(self.signal, 
                                fs=self.sampling_rate,
                                window=window,
                                nperseg=nperseg,
                                noverlap=noverlap,
                                return_onesided=False)
        
        # Shift zero frequency to center
        f = fftshift(f)
        Zxx = fftshift(Zxx, axes=0)
        
        # Convert to Doppler velocity
        velocity_axis = f * self.wavelength / 2
        
        # Magnitude spectrogram
        spectrogram = np.abs(Zxx)
        
        return spectrogram, t, velocity_axis
    
    def plot_spectrogram(self, spectrogram, time_axis, velocity_axis, 
                        title="Micro-Doppler Spectrogram", db_range=60):
        """
        Plot spectrogram with proper scaling and labels
        """
        # Convert to dB
        spec_db = 20 * np.log10(spectrogram + 1e-10)
        spec_db -= np.max(spec_db)  # Normalize to 0 dB
        
        plt.figure(figsize=(14, 8))
        
        extent = [time_axis[0]*1000, time_axis[-1]*1000,
                 velocity_axis[0], velocity_axis[-1]]
        
        im = plt.imshow(spec_db, 
                       aspect='auto',
                       extent=extent,
                       cmap='jet',
                       vmin=-db_range, vmax=0,
                       origin='lower',
                       interpolation='bilinear')
        
        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Doppler Velocity (m/s)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, label='Magnitude (dB)')
        plt.axhline(0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        plt.grid(True, alpha=0.3, color='white', linewidth=0.5)
        plt.tight_layout()
        plt.show()

# Test the class
print("\n" + "="*60)
print("MicroDopplerRadar class defined successfully!")
print("="*60)
# %% [markdown]
"""
---
# 4. Example 1: Single Drone with Rotors

Let's simulate a quadcopter drone and see its micro-Doppler signature!
"""

# %%
def example_single_drone():
    """
    Simulate a drone with 4 rotors and visualize micro-Doppler
    """
    # Create radar
    radar = MicroDopplerRadar(
        f_c=10e9,           # 10 GHz X-band
        prf=10e3,           # 10 kHz PRF
        observation_time=0.5  # 500 ms observation
    )
    
    # Add quadcopter drone
    radar.add_target_with_rotors(
        range_m=1500,        # 1.5 km away
        velocity_ms=15,      # Moving toward radar at 15 m/s
        rcs=0.01,            # Small RCS (10 cm² = 0.01 m²)
        num_rotors=4,        # Quadcopter
        rotor_radius=0.15,   # 15 cm blades
        rpm=3000,            # 3000 RPM
        blade_rcs_fraction=0.1  # Blades are 10% of body RCS
    )
    
    # Add noise
    radar.add_noise(snr_db=15)
    
    # Generate signal
    radar.generate_signal()
    
    # Generate spectrogram
    spec, t_axis, v_axis = radar.generate_spectrogram(
        nperseg=256,      # Frequency resolution
        noverlap=200      # Time resolution
    )
    
    # Plot
    radar.plot_spectrogram(spec, t_axis, v_axis,
                          title="Quadcopter Drone - Micro-Doppler Signature",
                          db_range=50)
    
    # Analysis
    print("\n" + "="*60)
    print("WHAT TO LOOK FOR IN THE SPECTROGRAM:")
    print("="*60)
    print("✓ Horizontal line at ~15 m/s: Body Doppler (main velocity)")
    print("✓ Sinusoidal modulation around body Doppler: Blade modulation")
    print("✓ Modulation bandwidth: ±(blade tip velocity)")
    print("✓ Periodicity: 4 blade flashes per rotation")
    print("\nExpected blade flash rate: 4 × 50 Hz = 200 Hz")
    print("="*60)

# Run example
example_single_drone()

# %% [markdown]
"""
## Understanding the Spectrogram

**What you're seeing:**

1. **Horizontal streak at +15 m/s:** This is the body Doppler
   - Constant velocity of the drone moving toward the radar
   
2. **Sinusoidal oscillations:** These are the micro-Doppler signatures
   - Width of oscillation = 2 × (blade tip velocity)
   - Frequency of oscillation = blade flash rate
   
3. **Periodic pattern:** 
   - For 4 blades, you see 4 "flashes" per rotation
   - Creates characteristic 200 Hz modulation

**Why this matters:**
- This signature is UNIQUE to rotating objects
- Birds don't have this regular, periodic pattern
- This is what makes drone detection possible!

---
# 5. Example 2: Drone vs Bird Comparison

Now let's compare drone and bird signatures side-by-side.
"""

# %%
def example_drone_vs_bird():
    """
    Compare micro-Doppler signatures of drone vs bird
    """
    # Scenario 1: Drone
    print("\n" + "="*60)
    print("SCENARIO 1: QUADCOPTER DRONE")
    print("="*60)
    
    radar_drone = MicroDopplerRadar(
        f_c=10e9,
        prf=10e3,
        observation_time=0.5
    )
    
    radar_drone.add_target_with_rotors(
        range_m=1500,
        velocity_ms=15,
        rcs=0.01,
        num_rotors=4,
        rotor_radius=0.15,
        rpm=3000,
        blade_rcs_fraction=0.15
    )
    
    radar_drone.add_noise(snr_db=12)
    radar_drone.generate_signal()
    
    spec_drone, t_drone, v_drone = radar_drone.generate_spectrogram(
        nperseg=256, noverlap=200
    )
    
    # Scenario 2: Bird
    print("\n" + "="*60)
    print("SCENARIO 2: BIRD")
    print("="*60)
    
    radar_bird = MicroDopplerRadar(
        f_c=10e9,
        prf=10e3,
        observation_time=0.5
    )
    
    radar_bird.add_target_with_wings(
        range_m=800,
        velocity_ms=8,
        rcs=0.005,
        wing_beat_freq=8,    # 8 Hz wing beats
        wing_length=0.25,    # 25 cm wings
        irregularity=0.4     # Irregular wing motion
    )
    
    radar_bird.add_noise(snr_db=12)
    radar_bird.generate_signal()
    
    spec_bird, t_bird, v_bird = radar_bird.generate_spectrogram(
        nperseg=256, noverlap=200
    )
    
    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Drone spectrogram
    spec_drone_db = 20 * np.log10(spec_drone + 1e-10)
    spec_drone_db -= np.max(spec_drone_db)
    
    extent_drone = [t_drone[0]*1000, t_drone[-1]*1000,
                   v_drone[0], v_drone[-1]]
    
    im1 = axes[0].imshow(spec_drone_db,
                        aspect='auto',
                        extent=extent_drone,
                        cmap='jet',
                        vmin=-50, vmax=0,
                        origin='lower',
                        interpolation='bilinear')
    
    axes[0].set_xlabel('Time (ms)', fontsize=12)
    axes[0].set_ylabel('Doppler Velocity (m/s)', fontsize=12)
    axes[0].set_title('DRONE: Regular, Periodic Blade Flashes', 
                     fontsize=13, fontweight='bold', color='darkgreen')
    axes[0].axhline(0, color='white', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, color='white')
    plt.colorbar(im1, ax=axes[0], label='dB')
    
    # Add annotation
    axes[0].text(0.05, 0.95, 'CHARACTERISTIC:\n• Periodic\n• 4 flashes/cycle\n• ~200 Hz rate',
                transform=axes[0].transAxes,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                verticalalignment='top', fontsize=10, fontweight='bold')
    
    # Bird spectrogram
    spec_bird_db = 20 * np.log10(spec_bird + 1e-10)
    spec_bird_db -= np.max(spec_bird_db)
    
    extent_bird = [t_bird[0]*1000, t_bird[-1]*1000,
                  v_bird[0], v_bird[-1]]
    
    im2 = axes[1].imshow(spec_bird_db,
                        aspect='auto',
                        extent=extent_bird,
                        cmap='jet',
                        vmin=-50, vmax=0,
                        origin='lower',
                        interpolation='bilinear')
    
    axes[1].set_xlabel('Time (ms)', fontsize=12)
    axes[1].set_ylabel('Doppler Velocity (m/s)', fontsize=12)
    axes[1].set_title('BIRD: Irregular Wing Beats',
                     fontsize=13, fontweight='bold', color='darkblue')
    axes[1].axhline(0, color='white', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, color='white')
    plt.colorbar(im2, ax=axes[1], label='dB')
    
    # Add annotation
    axes[1].text(0.05, 0.95, 'CHARACTERISTIC:\n• Irregular\n• Variable amplitude\n• ~8-16 Hz rate',
                transform=axes[1].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                verticalalignment='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("KEY DIFFERENCES:")
    print("="*60)
    print("DRONE:")
    print("  ✓ Regular, periodic modulation")
    print("  ✓ High frequency (~200 Hz blade flash)")
    print("  ✓ Consistent amplitude")
    print("  ✓ 4 distinct peaks per rotation")
    print("\nBIRD:")
    print("  ✓ Irregular, aperiodic modulation")
    print("  ✓ Low frequency (~8 Hz wing beat)")
    print("  ✓ Variable amplitude")
    print("  ✓ Non-uniform pattern")
    print("="*60)

# Run comparison
example_drone_vs_bird()

# %% [markdown]
"""
---
# 6. Feature Extraction for Classification

Now that we can see the differences visually, let's extract quantitative features
that can be used for automatic classification.

## Key Features

1. **Blade Flash Frequency** - Dominant frequency in the modulation
2. **Modulation Depth** - Peak-to-peak variation
3. **Periodicity Measure** - How regular is the pattern?
4. **Spectral Bandwidth** - Width of the modulation

These features can feed into:
- Classical ML: Random Forest, SVM
- Deep Learning: CNN on spectrograms (Part 6)
"""

# %%
def extract_micro_doppler_features(radar, spectrogram, time_axis, velocity_axis):
    """
    Extract quantitative features from micro-Doppler spectrogram
    
    Returns:
    --------
    features : dict
        Dictionary of extracted features
    """
    features = {}
    
    # 1. Find dominant velocity (body Doppler)
    velocity_profile = np.sum(spectrogram, axis=1)
    dominant_velocity_idx = np.argmax(velocity_profile)
    features['body_velocity'] = velocity_axis[dominant_velocity_idx]
    
    # 2. Extract velocity centroid over time (micro-Doppler track)
    velocity_centroid = np.zeros(len(time_axis))
    for t_idx in range(len(time_axis)):
        weights = spectrogram[:, t_idx]
        if np.sum(weights) > 0:
            velocity_centroid[t_idx] = np.sum(velocity_axis * weights) / np.sum(weights)
    
    # 3. Modulation depth (peak-to-peak)
    modulation = velocity_centroid - features['body_velocity']
    features['modulation_depth'] = np.max(modulation) - np.min(modulation)
    
    # 4. Blade flash frequency (FFT of modulation)
    # Remove mean
    modulation_ac = modulation - np.mean(modulation)
    
    # FFT to find periodicity
    fft_mod = np.fft.fft(modulation_ac)
    freq_mod = np.fft.fftfreq(len(modulation_ac), d=np.mean(np.diff(time_axis)))
    
    # Positive frequencies only
    pos_freq_idx = freq_mod > 0
    fft_mod_pos = np.abs(fft_mod[pos_freq_idx])
    freq_mod_pos = freq_mod[pos_freq_idx]
    
    # Find dominant frequency (blade flash rate)
    # Ignore very low frequencies (< 5 Hz)
    valid_freq_idx = freq_mod_pos > 5
    if np.any(valid_freq_idx):
        peak_idx = np.argmax(fft_mod_pos[valid_freq_idx])
        features['blade_flash_freq'] = freq_mod_pos[valid_freq_idx][peak_idx]
    else:
        features['blade_flash_freq'] = 0
    
    # 5. Periodicity measure (ratio of peak to mean in FFT)
    if len(fft_mod_pos[valid_freq_idx]) > 0:
        features['periodicity_ratio'] = (np.max(fft_mod_pos[valid_freq_idx]) / 
                                        np.mean(fft_mod_pos[valid_freq_idx]))
    else:
        features['periodicity_ratio'] = 1.0
    
    # 6. Spectral bandwidth (std of velocity distribution)
    features['spectral_bandwidth'] = np.std(velocity_centroid)
    
    return features

# %%
def demonstrate_feature_extraction():
    """
    Show feature extraction for drone and bird
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTION DEMONSTRATION")
    print("="*60)
    
    # Drone features
    radar_drone = MicroDopplerRadar(f_c=10e9, prf=10e3, observation_time=1.0)
    radar_drone.add_target_with_rotors(
        range_m=1500, velocity_ms=15, rcs=0.01,
        num_rotors=4, rotor_radius=0.15, rpm=3000
    )
    radar_drone.add_noise(snr_db=15)
    radar_drone.generate_signal()
    spec_drone, t_drone, v_drone = radar_drone.generate_spectrogram(nperseg=256, noverlap=200)
    
    features_drone = extract_micro_doppler_features(radar_drone, spec_drone, t_drone, v_drone)
    
    # Bird features
    radar_bird = MicroDopplerRadar(f_c=10e9, prf=10e3, observation_time=1.0)
    radar_bird.add_target_with_wings(
        range_m=800, velocity_ms=8, rcs=0.005,
        wing_beat_freq=8, wing_length=0.25, irregularity=0.4
    )
    radar_bird.add_noise(snr_db=15)
    radar_bird.generate_signal()
    spec_bird, t_bird, v_bird = radar_bird.generate_spectrogram(nperseg=256, noverlap=200)
    
    features_bird = extract_micro_doppler_features(radar_bird, spec_bird, t_bird, v_bird)
    
    # Display comparison
    print("\nDRONE FEATURES:")
    print("-" * 60)
    for key, value in features_drone.items():
        print(f"  {key:25s}: {value:8.2f}")
    
    print("\nBIRD FEATURES:")
    print("-" * 60)
    for key, value in features_bird.items():
        print(f"  {key:25s}: {value:8.2f}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION INSIGHTS:")
    print("="*60)
    print("✓ Blade flash freq:  Drone >> Bird  (200 Hz vs ~8 Hz)")
    print("✓ Periodicity ratio: Drone >> Bird  (regular vs irregular)")
    print("✓ Modulation depth:  Similar (both have moving parts)")
    print("\n→ These features can train a classifier (Random Forest, SVM, etc.)")
    print("="*60)

# Run demonstration
demonstrate_feature_extraction()

# %% [markdown]
"""
---
# 7. Multiple Targets Scenario

Real-world scenarios have multiple targets. Let's see how they appear together.
"""

# %%
def example_multiple_targets():
    """
    Simulate multiple targets: drone, bird, and aircraft
    """
    print("\n" + "="*60)
    print("COMPLEX SCENARIO: MULTIPLE TARGETS")
    print("="*60)
    
    radar = MicroDopplerRadar(
        f_c=10e9,
        prf=10e3,
        observation_time=0.8
    )
    
    # Target 1: Drone
    radar.add_target_with_rotors(
        range_m=1500,
        velocity_ms=15,
        rcs=0.01,
        num_rotors=4,
        rotor_radius=0.15,
        rpm=3000
    )
    
    # Target 2: Bird
    radar.add_target_with_wings(
        range_m=800,
        velocity_ms=-5,  # Moving away
        rcs=0.005,
        wing_beat_freq=10,
        wing_length=0.20,
        irregularity=0.3
    )
    
    # Target 3: Another drone (different RPM)
    radar.add_target_with_rotors(
        range_m=2000,
        velocity_ms=25,
        rcs=0.015,
        num_rotors=4,
        rotor_radius=0.12,
        rpm=4000  # Faster rotors
    )
    
    radar.add_noise(snr_db=12)
    radar.generate_signal()
    
    spec, t_axis, v_axis = radar.generate_spectrogram(nperseg=256, noverlap=220)
    
    radar.plot_spectrogram(spec, t_axis, v_axis,
                          title="Multiple Targets: 2 Drones + 1 Bird",
                          db_range=50)
    
    print("\n" + "="*60)
    print("WHAT YOU'RE SEEING:")
    print("="*60)
    print("✓ Three distinct velocity tracks (horizontal streaks)")
    print("✓ Two show high-frequency modulation (drones at 200 Hz, 267 Hz)")
    print("✓ One shows low-frequency modulation (bird at ~10 Hz)")
    print("✓ Each target has unique micro-Doppler signature")
    print("="*60)

# Run example
example_multiple_targets()

# %% [markdown]
"""
---
# 8. Understanding STFT Parameters

## Trade-offs in Spectrogram Generation

The **Short-Time Fourier Transform (STFT)** has parameters that affect resolution:

### 1. Window Length (nperseg)
- **Longer window** → Better frequency resolution, worse time resolution
- **Shorter window** → Better time resolution, worse frequency resolution
- **Typical values:** 128 - 512 samples

### 2. Overlap (noverlap)
- **More overlap** → Smoother spectrogram, more computation
- **Less overlap** → Faster computation, choppier appearance
- **Typical:** 50% - 75% overlap (nperseg//2 to 3*nperseg//4)

### 3. Window Function
- **Hann:** Good general purpose, reduces spectral leakage
- **Hamming:** Slightly better frequency resolution
- **Blackman:** Best spectral leakage suppression, wider main lobe

Let's see the impact visually!
"""

# %%
def compare_stft_parameters():
    """
    Show how STFT parameters affect spectrogram quality
    """
    # Create drone signal
    radar = MicroDopplerRadar(f_c=10e9, prf=10e3, observation_time=0.5)
    radar.add_target_with_rotors(
        range_m=1500, velocity_ms=15, rcs=0.01,
        num_rotors=4, rotor_radius=0.15, rpm=3000
    )
    radar.add_noise(snr_db=15)
    radar.generate_signal()
    
    # Different window lengths
    window_sizes = [128, 256, 512]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, nperseg in enumerate(window_sizes):
        spec, t_axis, v_axis = radar.generate_spectrogram(
            nperseg=nperseg,
            noverlap=int(nperseg * 0.75)
        )
        
        spec_db = 20 * np.log10(spec + 1e-10)
        spec_db -= np.max(spec_db)
        
        extent = [t_axis[0]*1000, t_axis[-1]*1000, v_axis[0], v_axis[-1]]
        
        im = axes[idx].imshow(spec_db,
                             aspect='auto',
                             extent=extent,
                             cmap='jet',
                             vmin=-50, vmax=0,
                             origin='lower')
        
        axes[idx].set_xlabel('Time (ms)')
        axes[idx].set_ylabel('Velocity (m/s)')
        axes[idx].set_title(f'Window Length = {nperseg}', fontweight='bold')
        axes[idx].axhline(0, color='white', linestyle='--', alpha=0.5)
        axes[idx].grid(True, alpha=0.3, color='white')
        plt.colorbar(im, ax=axes[idx], label='dB')
        
        # Compute resolutions
        freq_res = radar.sampling_rate / nperseg
        time_res = nperseg / radar.sampling_rate * 1000  # ms
        
        axes[idx].text(0.05, 0.05, 
                      f'Freq res: {freq_res:.1f} Hz\nTime res: {time_res:.1f} ms',
                      transform=axes[idx].transAxes,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      verticalalignment='bottom', fontsize=9)
    
    plt.suptitle('Effect of Window Length on Spectrogram Quality', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("WINDOW LENGTH TRADE-OFFS:")
    print("="*60)
    print("SHORT window (128):")
    print("  ✓ Good time resolution (see rapid changes)")
    print("  ✗ Poor frequency resolution (smeared in frequency)")
    print("\nMEDIUM window (256):")
    print("  ✓ Balanced trade-off")
    print("  ✓ Good for most micro-Doppler analysis")
    print("\nLONG window (512):")
    print("  ✓ Excellent frequency resolution (sharp lines)")
    print("  ✗ Poor time resolution (slow updates)")
    print("="*60)

# Run comparison
compare_stft_parameters()

# %% [markdown]
"""
---
# 9. Practical Considerations

## Real-World Challenges

1. **SNR Variation**
   - Targets at different ranges have different SNRs
   - Micro-Doppler signatures degrade with low SNR
   - Need robust feature extraction

2. **Multi-Target Scenarios**
   - Spectrograms can overlap in velocity
   - Need clustering/separation algorithms
   - Track association becomes critical

3. **Environmental Effects**
   - Wind affects drone flight (velocity variation)
   - Rain/snow adds clutter
   - Ground clutter at v≈0

4. **Parameter Selection**
   - STFT window must match expected modulation rates
   - Too short: miss periodic structure
   - Too long: smear rapid changes

## Integration with RDM

**Remember from Part 2:**
- RDM shows WHERE targets are (range-velocity)
- Spectrogram shows WHAT targets are (time-frequency signature)

**Processing pipeline:**
1. Generate RDM → Detect targets (Part 4: CFAR)
2. For each detection → Extract range-velocity cell
3. Generate spectrogram for that cell
4. Extract micro-Doppler features
5. Classify (drone vs bird vs aircraft)
"""

# %% [markdown]
"""
---
# 10. Exercises & Experiments

Try these to deepen your understanding:

## Exercise 1: Rotor Count Effect
Create drones with 2, 3, 4, 6 blades:
- How does blade count affect the spectrogram?
- Can you identify the number of blades from the signature?

## Exercise 2: RPM Variation
Vary RPM from 1000 to 5000:
- How does modulation frequency change?
- At what RPM does the signature become hard to see?

## Exercise 3: SNR Threshold
Reduce SNR from 20 dB to 0 dB:
- At what SNR does micro-Doppler disappear?
- Which features are most robust to noise?

## Exercise 4: Window Optimization
For a blade flash rate of 200 Hz:
- What minimum window length do you need to resolve it?
- What happens if window is too short?

## Exercise 5: Feature-Based Classifier
Collect features from 20 drones and 20 birds:
- Train a simple classifier (threshold on blade_flash_freq)
- What accuracy can you achieve?
- Which features are most discriminative?

---
# 11. Key Takeaways

✅ **Micro-Doppler** = Time-varying Doppler from rotating/vibrating parts
✅ **Spectrograms** show time-frequency evolution (STFT)
✅ **Drones** have periodic, high-frequency blade modulation (~200 Hz)
✅ **Birds** have irregular, low-frequency wing beats (~8 Hz)
✅ **Features** can be extracted for classification:
   - Blade flash frequency
   - Modulation depth
   - Periodicity ratio
✅ **STFT parameters** trade off time vs frequency resolution
✅ **Integration with RDM** gives complete picture: WHERE + WHAT

---
# What's Next?

You now understand micro-Doppler signatures! In **Part 4**, we'll cover:

1. **CFAR Detection** - Automatically detect targets in RDM
2. **Classical CFAR** - CA-CFAR, OS-CFAR algorithms
3. **Neural CFAR** - Learned detection using CNNs
4. **Performance Analysis** - Pd vs Pfa curves

**Next Steps:**
1. Run all examples in this notebook
2. Try the exercises
3. Experiment with different parameters
4. Think about: "How would I build an automatic classifier?"

---
"""

# %%
# Notebook size check
import os
import sys

notebook_file = 'part3_micro_doppler.py'
if os.path.exists(notebook_file):
    size_bytes = os.path.getsize(notebook_file)
    size_kb = size_bytes / 1024
    
    print("\n" + "="*60)
    print(f"Notebook size: {size_kb:.1f} KB ({size_bytes:,} bytes)")
    if size_kb > 900:
        print("⚠️  WARNING: Approaching 1 MB limit!")
    else:
        print("✓ Size is good!")
    print("="*60)
else:
    print(f"File {notebook_file} not found")

print("\n" + "="*60)
print("PART 3: MICRO-DOPPLER SIGNATURES - COMPLETE!")
print("="*60)
print("\nCongratulations! You've completed Part 3.")
print("\nYou now understand:")
print("  ✓ What micro-Doppler is and why it matters")
print("  ✓ Physics of rotor blade modulation")
print("  ✓ How to generate and analyze spectrograms")
print("  ✓ Feature extraction for classification")
print("  ✓ Drone vs bird signature differences")
print("\nReady for Part 4: CFAR Detection!")
print("="*60)

