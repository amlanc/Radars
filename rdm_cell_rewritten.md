---
# 2. The Range-Doppler Map (RDM): Bringing It All Together

## The Core Idea: Why We Need 2D Processing

**Think back to basics:**
- **One pulse** → Process its samples → Get range (via time delay)
- **Multiple pulses** → Compare phase changes → Get velocity (via Doppler)

**The challenge:** A target has BOTH range AND velocity. How do we measure both simultaneously?

**The solution:** Organize data into a 2D matrix and process in two dimensions.

---

## How Pulse-Doppler Radar Collects Data

**Physical picture:**

Imagine you're transmitting 128 pulses, one every 100 μs (your PRI). Each pulse travels out, hits targets, and returns. You digitize each echo into 256 samples.

**What you end up with:**
```
Pulse 1:  [sample_0, sample_1, sample_2, ..., sample_255]  ← 256 range samples
Pulse 2:  [sample_0, sample_1, sample_2, ..., sample_255]
Pulse 3:  [sample_0, sample_1, sample_2, ..., sample_255]
...
Pulse 128: [sample_0, sample_1, sample_2, ..., sample_255]
```

**This creates a 2D data matrix: [128 pulses × 256 samples]**

---

## Understanding the Two Time Scales

**Fast-time (horizontal, within one pulse):**
- Time within a single pulse as it's being sampled
- Samples 0→255 happen in ~10 μs (your pulse duration)
- This encodes **range** (because time = distance / c)

**Slow-time (vertical, across pulses):**
- Time from pulse to pulse
- Pulses 0→127 span 12.8 ms (your CPI duration)
- This encodes **velocity** (because phase changes between pulses)

**Analogy:** Think of a music spectrogram
- Horizontal axis = time within each analysis window (like fast-time)
- Vertical axis = frequency bins showing how spectrum changes over time (like slow-time)
- RDM is similar: fast-time → range, slow-time → velocity

---

## Two-Stage Processing: Building the RDM

### Step 1: Range-FFT (Process Each Pulse Individually)

**What we do:**
```
For each pulse (each row):
    Take FFT across the 256 samples → Get 256 range bins
```

**What this reveals:**
- Converts time-domain samples → frequency-domain range bins
- Each pulse now shows "what's at what range"
- BUT we still have 128 pulses, so we have 128 copies of "range profile"

**After Step 1:** Matrix is now [128 pulses × 256 range bins]

### Step 2: Doppler-FFT (Process Across Pulses)

**What we do:**
```
For each range bin (each column):
    Take FFT across the 128 pulses → Get 128 Doppler bins
```

**What this reveals:**
- At range bin k, how did the signal change from pulse to pulse?
- Stationary target: no phase change → DC (zero Doppler)
- Moving target: linear phase change → frequency offset (Doppler shift)

**After Step 2:** Matrix is now [128 Doppler bins × 256 range bins] = **RDM**

---

## Visual Processing Flow

```
Raw I/Q Data Matrix               After Range-FFT                After Doppler-FFT
[128 pulses × 256 samples]  →  [128 pulses × 256 range bins]  →  [128 Doppler × 256 range]
      
  Time samples                     Range information              Range-Doppler Map (RDM)
  per pulse                        per pulse                      
                                                                  X-axis: Range
                                                                  Y-axis: Velocity
```

**Shortcut notation:**
```
RDM = FFT2D(raw_data_matrix)
```
This is mathematically equivalent to doing Range-FFT then Doppler-FFT.

---

## The Math Behind It (Now That You Understand the Concept)

### Signal Model

**For a single target at range R and velocity v:**

```
s[m, n] = A · exp(j · 2π · (f_range · n/f_sampling + f_doppler · m · PRI))

Where:
m = pulse index (0 to 127)          ← slow-time index
n = sample index (0 to 255)         ← fast-time index
f_range = 2·R·B/c                   ← range frequency (from round-trip delay)
f_doppler = 2·v/λ                   ← Doppler frequency (from motion)
```

**Key insight:** This is a 2D sinusoid!
- Oscillates in n-direction (fast-time) at frequency f_range
- Oscillates in m-direction (slow-time) at frequency f_doppler
- 2D FFT finds these two frequencies → gives you R and v

### After 2D FFT

```
RDM[doppler_bin, range_bin] = |FFT2D(s[m,n])|²

The target appears as a bright peak at:
- range_bin corresponding to its distance
- doppler_bin corresponding to its velocity
```

---

## Converting Bins to Physical Units

### Range Axis (X-axis)

Each range bin k corresponds to:
```
Range[k] = k · c / (2 · B)

Example: B = 150 MHz, k = 100
Range = 100 · (3×10⁸) / (2 · 150×10⁶) = 100 m
```

**Why?** Frequency bin spacing = B/N, and range = c·Δt/2 = c·Δf/(2·B)

### Velocity Axis (Y-axis)

Each Doppler bin m corresponds to:
```
Velocity[m] = (m - M/2) · λ · PRF / (2 · M)

Where:
- M = number of pulses (128)
- (m - M/2) shifts zero velocity to center
- PRF = 1/PRI (pulse repetition frequency)

Example: λ = 3 cm, PRF = 10 kHz, M = 128, m = 80
Velocity = (80 - 64) · 0.03 · 10000 / (2 · 128) = 18.75 m/s (approaching)
```

**Why?** Doppler bin spacing = PRF/M, and velocity = f_doppler · λ/2

---

## Quick Reality Check

**What you should now understand:**
1. ✅ Why we need a 2D matrix (two parameters: range and velocity)
2. ✅ What fast-time and slow-time mean physically
3. ✅ Why Range-FFT first (process each pulse → get range)
4. ✅ Why Doppler-FFT second (compare pulses → get velocity)
5. ✅ How bins map to real-world range and velocity

**Next section shows you actual RDM images and what to look for!**
