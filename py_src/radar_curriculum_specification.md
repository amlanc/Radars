<!-- Copyright © 2025 Amlan Chatterjee. All rights reserved. -->

# RADAR SIGNAL PROCESSING TUTORIAL CURRICULUM SPECIFICATION

# Complete Guide for a Tutorial Creation from Scratch

**Document Version:** 1.0
**Date:** October 10, 2025
**Purpose:** Self-contained specification to recreate complete tutorial series

---

## TABLE OF CONTENTS

1. Student Profile & Context
2. Overall Learning Architecture
3. Teaching Style & Formatting Standards
4. Technical Requirements & Constraints
5. Existing Foundation (Parts 1-2)
6. Detailed Part Specifications (Parts 3-8)
7. Code Quality Standards
8. Visualization Guidelines
9. Size Management Strategy
10. Assessment & Validation Criteria

---

## 1. STUDENT PROFILE & CONTEXT

### Background

- **Education:** Electronics engineer, studied radar systems 30 years ago
- **Current Skills:** Strong Python programmer, experienced in AI/ML (25+ years software development)
- **Weakness:** Frontend development, rusty on radar fundamentals
- **Learning Style:** Hands-on, code-first, needs physical intuition before mathematics and mathematics intuition before code

### Professional Context

- Starting new role in counter-drone cognitive radar systems
- Needs practical, immediately applicable knowledge
- Timeline: 1 weeks to get productive
  **CRITICAL:** No company names in code/notebooks (no "chaos", etc.)

### End Goal

Build toward **cognitive radar** - autonomous, multi-agent system where:

- Each processing stage is an intelligent agent
- Agents make independent decisions (filter selection, thresholds, classification)
- System self-tunes based on environment
- Uses AI to replace classical signal processing where beneficial

---

## 2. OVERALL LEARNING ARCHITECTURE

### Pedagogical Approach: "Hybrid Classical-AI"

For each major topic:

1. Teach classical signal processing method
2. Explain why it works physically
3. Show its limitations
4. Introduce AI/ML replacement
5. Compare performance
6. Discuss when to use each

### Tutorial Progression

**FOUNDATION (Already Complete):**

- Part 1: Radar Fundamentals (Range-FFT, I/Q, basic parameters)
- Part 2: Doppler Processing & RDM generation

**CORE CAPABILITIES (Parts 3-6):**

- Part 3: Micro-Doppler Signatures (Classical physics + feature extraction)
- Part 4: CFAR Detection (Classical algorithms â†’ Learned detection)
- Part 5: Clutter Suppression (MTI/STAP â†’ Neural clutter cancellation)
- Part 6: Classification (Feature-based â†’ CNN/Transformer)

**SYSTEM INTEGRATION (Parts 7-8):**

- Part 7: Tracking (Kalman â†’ LSTM/Attention mechanisms)
- Part 8: Cognitive Radar Framework (Multi-agent architecture)

### Priority Levels

**Tier 1 (Critical - Must complete):** Parts 3, 4, 6
**Tier 2 (Important):** Parts 5, 8
**Tier 3 (If time allows):** Part 7

---

## 3. TEACHING STYLE & FORMATTING STANDARDS

### Core Principles

**From student's preferred style (see rdm_cell_rewritten.md):**

1. **Physical Intuition First, Math Second**

   - Start with "Think back to basics" or "The Core Idea"
   - Use analogies (music spectrogram, ripples in pond, etc.)
   - Explain WHY physically before showing equations

2. **Structure: Concept â†’ Intuition Check â†’ Formula â†’ Code â†’ Experiment**

   - Markdown explanation of concept (physical intuition)
   - **Intuition-testing questions** (verify understanding before proceeding)
   - Mathematical formulation (only after intuition is solid)
   - Working code implementation
   - Interactive examples to experiment with

3. **Reality Checks**

   - End each major section with "What you should now understand"
   - Use checkbox lists (âœ… symbols)
   - Include quick self-check questions

4. **Visual Thinking**

   - Heavy use of ASCII diagrams in markdown
   - Flow diagrams showing data transformations
   - Multiple subplots showing different views of same data

5. **Conversational Tone**

   - Address reader directly ("You now understand...", "Think about...")
   - Anticipate questions ("You might wonder why...")
   - Build confidence ("This is the key insight!")

6. **Intuition Testing Before Math**
   - After physical explanation, ALWAYS include 2-4 questions
   - Questions should test conceptual understanding, not calculation
   - Provide answers/explanations after questions
   - Only proceed to formulas after intuition is verified

### Intuition-Testing Question Guidelines

**Purpose:** Verify student grasps physical concept before introducing math

**Question Types:**

1. **Predict behavior:** "What happens if we double the PRF?"
2. **Compare scenarios:** "Which target will have higher Doppler: one at 20 m/s or one at 40 m/s?"
3. **Explain physically:** "Why does a stationary target create a vertical line in the RDM?"
4. **Edge cases:** "What if the rotor stops spinning - what happens to micro-Doppler?"
5. **Reasoning:** "If you see periodic modulation in a spectrogram, what does that tell you about the target?"

**Format:**

```markdown
---
## ðŸ¤” Test Your Intuition

Before we dive into the math, let's make sure the physical concept is clear.

**Question 1:** A drone is moving at 15 m/s toward the radar. If the drone stops moving but its rotors keep spinning, what happens to:
- a) The macro-Doppler shift?
- b) The micro-Doppler signature?

**Question 2:** You see two targets at the same range in an RDM. Target A appears as a single bright spot. Target B appears as a vertical bright line. What can you conclude about each target's motion?

**Question 3:** [Scenario-based question]

<details>
<summary><b>ðŸ’¡ Click to see answers and explanations</b></summary>

**Answer 1:**
- a) Macro-Doppler goes to zero (no bulk motion)
- b) Micro-Doppler remains (rotors still spinning!)

**Why?** Macro-Doppler comes from the target's center-of-mass velocity. Micro-Doppler comes from rotating parts. They're independent!

**Answer 2:**
- Target A is moving (has specific velocity â†’ single Doppler bin)
- Target B is stationary (velocity = 0 â†’ all energy at zero Doppler)

**Why?** Stationary targets have the same phase across all pulses, so they appear at v=0 across all range bins.

**Answer 3:** [Detailed explanation]

</details>

---
```

**Bad questions (avoid):**

- âŒ "Calculate the Doppler frequency for v=20 m/s" (this is math, not intuition)
- âŒ "What is the formula for range resolution?" (rote memorization)
- âŒ "True or false: PRF stands for Pulse Repetition Frequency" (trivial)

**Good questions:**

- âœ… "If you increase bandwidth, will you be able to separate targets that are closer together or farther apart?"
- âœ… "A bird and a drone are both at 1000m moving at 15 m/s. Can you tell them apart from the RDM alone? If not, what additional processing do you need?"
- âœ… "Sketch what you expect the spectrogram to look like for a 4-bladed drone vs a 2-bladed drone"

### Markdown Cell Structure Template

```markdown
---
# [Section Number]. [Section Title]: [Subtitle]

## The Core Idea / Physical Picture

[Start with intuition, analogy, or physical explanation]

**Think back to basics:**
- [Connection to prior knowledge]
- [Build up from fundamentals]

---

## ðŸ¤” Test Your Intuition

Before we move to formulas, verify you understand the concept:

**Question 1:** [Predict behavior or compare scenarios]

**Question 2:** [Explain physically what's happening]

**Question 3:** [Edge case or reasoning question]

<details>
<summary><b>ðŸ’¡ Click to see answers</b></summary>

**Answer 1:** [Explanation with physical reasoning]

**Answer 2:** [Explanation with physical reasoning]

**Answer 3:** [Explanation with physical reasoning]

</details>

---

## [Subsection]: The Math Behind It

**[Concept name]:**
```

[Formula with clear variable definitions]

Where:

- variable = explanation [units]
- ...

```

**Why this formula?** [Physical explanation connecting back to intuition]

---

## Quick Reality Check

**What you should now understand:**
1. âœ… [Concept 1]
2. âœ… [Concept 2]
...

**Next: [Preview of next section]**
```

### Code Cell Structure Template

```python
def descriptive_function_name(parameters):
    """
    Clear description of what this does

    Parameters:
    - param1: description [units]
    - param2: description [units]

    Returns:
    - output1: description [units]
    """
    # Step 1: [What we're doing]
    code_step_1

    # Step 2: [What we're doing]
    code_step_2

    # Visualization
    fig, axes = plt.subplots(...)
    [plotting code]
    plt.tight_layout()
    plt.show()

    return results

# Example usage (commented out by default)
# example_call()
```

### Formatting Rules

- Use `**bold**` for emphasis on key terms (first use only)
- Use `code formatting` for variable names in text
- Equations in markdown code blocks with clear variable definitions
- No excessive emoji (âœ… for checklists is OK)
- Headers: # for main, ## for subsections, ### for sub-subsections
- Always include units in brackets [m], [Hz], [m/s]

---

## 4. TECHNICAL REQUIREMENTS & CONSTRAINTS

### File Size Management (CRITICAL)

**Hard limit: Each notebook < 1 MB**

**Strategies to stay under limit:**

1. No inline base64 images in notebooks
2. Save plots to files, don't embed
3. Limit output cell sizes
4. Use `plt.show()` instead of inline display when possible
5. Keep examples focused (don't generate massive arrays)
6. If approaching limit: split into Part 3a, Part 3b

**Size check before delivery:**

```python
import os
notebook_size = os.path.getsize('notebook.ipynb')
assert notebook_size < 1_000_000, f"Notebook too large: {notebook_size/1e6:.2f} MB"
```

### Code Quality Standards

- **Defensive programming:** Input validation, error handling
- **Production quality:** No TODO, no placeholders, no mock data
- **Comments:** Explain WHY, not WHAT
- **Type hints:** Use where it aids clarity
- **Docstrings:** Google-style for all functions/classes

### Python Environment

- Python 3.10+
- Standard libraries: numpy, scipy, matplotlib
- ML libraries: torch, tensorflow (when needed in later parts)
- ipywidgets: For interactive intuition questions (check before install)
- No exotic dependencies

### Interactive Questions Setup Pattern

**Include at start of each notebook:**

```python
# Run this cell first - sets up interactive questions
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML
    print("âœ… ipywidgets already installed")
except ImportError:
    print("ðŸ“¦ Installing ipywidgets (one-time setup)...")
    %pip install ipywidgets -q
    import ipywidgets as widgets
    from IPython.display import display, HTML
    print("âœ… Installation complete!")

def create_intuition_check(question_text, options, correct_idx,
                          explanation_correct, explanation_wrong):
    """
    Creates embedded multiple choice question with immediate feedback

    Parameters:
    - question_text: The question to ask
    - options: List of answer choices
    - correct_idx: Index of correct answer (0-based)
    - explanation_correct: Explanation shown when correct
    - explanation_wrong: Hint/explanation shown when incorrect
    """
    # Styling
    style = """
    <style>
        .question-container {
            background-color: #f8f9fa;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .question-text {
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 10px;
            color: #2c3e50;
        }
    </style>
    """

    # Radio buttons
    radio = widgets.RadioButtons(
        options=options,
        layout=widgets.Layout(width='100%', margin='10px 0'),
        style={'description_width': '0px'}
    )

    # Check button
    check_btn = widgets.Button(
        description='Check Answer',
        button_style='',
        layout=widgets.Layout(width='150px', margin='10px 0')
    )

    # Feedback area
    feedback = widgets.HTML(layout=widgets.Layout(margin='10px 0'))

    def on_check(b):
        if radio.value == options[correct_idx]:
            feedback.value = f"""
            <div style='background-color: #d4edda; border: 1px solid #c3e6cb;
                        padding: 12px; border-radius: 4px; color: #155724;'>
                <b>âœ… Correct!</b><br><br>
                {explanation_correct}
            </div>
            """
            check_btn.button_style = 'success'
        else:
            feedback.value = f"""
            <div style='background-color: #f8d7da; border: 1px solid #f5c6cb;
                        padding: 12px; border-radius: 4px; color: #721c24;'>
                <b>âŒ Not quite.</b><br><br>
                {explanation_wrong}
            </div>
            """
            check_btn.button_style = 'danger'

    check_btn.on_click(on_check)

    # Display
    display(HTML(style))
    display(HTML(f'<div class="question-container"><div class="question-text">{question_text}</div></div>'))
    display(radio)
    display(check_btn)
    display(feedback)

print("âœ… Interactive questions ready!")
```

**Usage in notebook:**

```python
# After physical explanation, before formulas
create_intuition_check(
    question_text="A drone is moving at 15 m/s. If it stops but rotors keep spinning, what happens?",
    options=[
        "Both macro and micro-Doppler go to zero",
        "Macro-Doppler goes to zero, micro-Doppler remains",
        "Both stay the same",
        "Macro-Doppler remains, micro-Doppler goes to zero"
    ],
    correct_idx=1,
    explanation_correct="""
        <b>Why:</b> Macro-Doppler = 2v/Î» where v is bulk velocity (now 0).
        Micro-Doppler comes from rotor rotation (still spinning!). They're independent.
    """,
    explanation_wrong="""
        Think about what creates each:<br>
        â€¢ Macro: Bulk motion of target<br>
        â€¢ Micro: Rotating/vibrating parts<br>
        What changed? What stayed the same?
    """
)
```

### Visualization Standards

```python
# Standard setup (include in each notebook)
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
```

**Color schemes:**

- RDMs: 'jet' colormap (industry standard)
- Spectrograms: 'viridis' or 'magma'
- Time-series: distinct colors with labels
- Use `alpha=0.3` for grid lines

**Plot requirements:**

- Always label axes with units
- Include titles with clear descriptions
- Use `plt.tight_layout()` before show/save
- Save figures with `dpi=150, bbox_inches='tight'`
- Grid: `grid(True, alpha=0.3)`

---

## 5. EXISTING FOUNDATION (Parts 1-2)

### Part 1: Radar Fundamentals

**File:** `radar_fundamentals_notebook.py` (converted to .ipynb)

**Key concepts covered:**

- FMCW vs Pulse-Doppler comparison
- I/Q signal representation
- Radar equation
- Range-FFT processing
- Resolution limits

**Key classes/functions:**

- `RadarParams` class: Calculates system parameters
- `visualize_iq_signal()`: Demonstrates I/Q representation
- `generate_pulse_return()`: Simulates received signal
- `range_fft_processing()`: Single-pulse range processing

**Student should understand:**

- âœ… How I/Q preserves phase information
- âœ… Why Range-FFT converts time â†’ range bins
- âœ… Trade-offs in radar parameters (PRF, bandwidth)

### Part 2: Doppler Processing & RDM

**File:** `radar_doppler_rdm_notebook.py/.ipynb`

**Key concepts covered:**

- Doppler shift physics (why factor of 2)
- Phase progression across pulses
- Fast-time vs slow-time
- 2D FFT creating RDM
- MTI filtering for clutter suppression

**Key class: PulseDopplerRadar**

```python
class PulseDopplerRadar:
    def __init__(self, f_c=10e9, PRF=10e3, bandwidth=150e6,
                 num_pulses=128, samples_per_pulse=512)
    def add_target(self, range_m, velocity_ms, rcs=1.0)
    def add_noise(self, snr_db=20)
    def generate_rdm(self, window='hamming')
    def plot_rdm(self, rdm, range_axis, velocity_axis, ...)
```

**Student should understand:**

- âœ… Why stationary targets appear as vertical line at v=0
- âœ… Why moving targets appear as single bright spot
- âœ… How MTI removes DC component
- âœ… Resolution in range (from bandwidth) and velocity (from CPI length)

**Critical: Parts 3+ must build on this PulseDopplerRadar class**

---

## 6. DETAILED PART SPECIFICATIONS

---

### PART 3: MICRO-DOPPLER SIGNATURES

**Learning Objectives:**

1. Understand physical mechanism of rotor blade modulation
2. Generate time-frequency spectrograms (Short-Time Fourier Transform)
3. Extract micro-Doppler features
4. Distinguish drone signatures from bird signatures

**Prerequisites:** Parts 1-2, understand RDM generation

**Estimated Time:** 2-3 days to complete

#### Section 1: Physical Mechanism of Micro-Doppler

**Concept to teach:**

- Macro-Doppler: Bulk motion of target center-of-mass
- Micro-Doppler: Additional modulation from rotating/vibrating parts
- For drones: Rotor blades create periodic frequency modulation

**Physical intuition:**

```
Drone body moving at 15 m/s â†’ Doppler shift = 1000 Hz (macro)
Rotor blade tip moving at 50 m/s (tangential) â†’ Additional Â±3333 Hz (micro)
Result: Signal oscillates between 1000-3333 = -2333 Hz and 1000+3333 = 4333 Hz
Frequency: Modulation rate = rotor RPM
```

**Math to include:**

```
Rotor blade micro-Doppler:
f_micro(t) = (4Ï€ / Î») * r_blade * Ï‰ * sin(Ï‰*t + Ï†)

Where:
- r_blade = blade radius [m]
- Ï‰ = angular velocity [rad/s] = 2Ï€ * RPM / 60
- Ï† = blade phase angle
```

**Visualization needed:**

1. Diagram showing blade rotation and instantaneous velocity vectors
2. Time-domain signal showing amplitude modulation
3. Comparison: drone (with micro-Doppler) vs aircraft (without)

**Key insight to emphasize:**
"The rotor blades act like a FREQUENCY MODULATOR on top of the bulk Doppler shift. This creates sidebands that are unique to drones!"

#### Section 2: Time-Frequency Analysis (Spectrograms)

**Concept to teach:**

- RDM shows range-velocity at one snapshot
- Micro-Doppler is TIME-VARYING - need to see how Doppler changes over time
- Solution: Short-Time Fourier Transform (STFT)

**Analogy:**
"RDM is like a photograph - shows where things are at one instant. Spectrogram is like a video - shows how the Doppler signature evolves."

**Implementation:**

```python
class MicroDopplerRadar(PulseDopplerRadar):
    """Extends PulseDopplerRadar with rotor blade simulation"""

    def add_drone_with_rotors(self, range_m, velocity_ms,
                              rpm, num_blades, blade_length, rcs=1.0):
        """
        Add drone with rotating blades

        Parameters:
        - rpm: Rotor revolutions per minute
        - num_blades: Number of blades (typically 2-4)
        - blade_length: Blade radius [m]
        """
        # Generate body return
        # Add blade flash modulation
        # Phase offsets for multiple blades

    def generate_spectrogram(self, range_bin, window_length=32, overlap=0.75):
        """
        Generate micro-Doppler spectrogram for specific range bin

        Returns:
        - spectrogram: Time-frequency representation
        - time_axis: Time values [s]
        - freq_axis: Doppler frequency [Hz]
        """
        # Extract signal at range_bin across all pulses
        # Apply STFT with sliding window
        # Return magnitude spectrogram
```

**Visualization needed:**

1. Spectrogram showing blade flash signatures (bright lines modulating up/down)
2. Comparison: Bird (random flutter) vs Drone (periodic modulation)
3. Multiple blades creating multiple modulation lines

**Parameters for examples:**

```python
# Typical drone
rpm = 3000  # 50 Hz blade flash
num_blades = 4
blade_length = 0.15  # 15 cm radius

# Typical bird
wing_beat_freq = 8  # Hz (much slower, irregular)
```

#### Section 3: Feature Extraction from Micro-Doppler

**Concept to teach:**
Extract quantitative features from spectrograms to feed into classifiers

**Features to implement:**

1. **Blade Flash Frequency (BFF):** Peak in frequency vs time plot
2. **Modulation Depth:** Max - Min Doppler over time
3. **Periodicity:** Auto-correlation of Doppler centroid
4. **Bandwidth:** Spread of micro-Doppler signature

**Code structure:**

```python
def extract_micro_doppler_features(spectrogram, time_axis, freq_axis):
    """
    Extract features for classification

    Returns:
    - features: Dictionary of feature values
    """
    features = {}

    # 1. Blade flash frequency
    doppler_centroid = compute_centroid(spectrogram, freq_axis)
    fft_centroid = fft(doppler_centroid)
    features['blade_flash_freq'] = find_dominant_freq(fft_centroid)

    # 2. Modulation depth
    features['mod_depth'] = np.max(doppler_centroid) - np.min(doppler_centroid)

    # 3. Periodicity measure
    autocorr = np.correlate(doppler_centroid, doppler_centroid, mode='full')
    features['periodicity'] = measure_periodicity(autocorr)

    # 4. Bandwidth
    features['bandwidth'] = np.std(spectrogram, axis=0).mean()

    return features
```

#### Section 4: Practical Examples

**Scenario 1: Single Drone**

```python
radar = MicroDopplerRadar(num_pulses=256, samples_per_pulse=512)
radar.add_drone_with_rotors(
    range_m=1500,
    velocity_ms=15,
    rpm=3000,
    num_blades=4,
    blade_length=0.15,
    rcs=0.01
)
radar.add_noise(snr_db=10)

# Generate spectrogram at drone's range bin
spec, t_axis, f_axis = radar.generate_spectrogram(range_bin=...)
plot_spectrogram(spec, t_axis, f_axis)
```

**Scenario 2: Drone vs Bird Comparison**

```python
# Create two separate simulations
# Extract features from each
# Show clear difference in blade flash frequency
# Bird: irregular, low frequency flutter
# Drone: periodic, high frequency modulation
```

#### Exercises for Part 3

1. **Blade count experiment:** Generate drones with 2, 3, 4 blades. How does spectrogram change?
2. **RPM variation:** What happens at 1000 RPM vs 5000 RPM?
3. **SNR threshold:** At what SNR does micro-Doppler signature become undetectable?
4. **Feature robustness:** Add noise, see which features remain stable

#### Reality Check for Part 3

**What you should now understand:**

1. âœ… Micro-Doppler is caused by rotating/vibrating parts on target
2. âœ… Spectrograms show time-varying Doppler (RDM cannot show this)
3. âœ… Blade flash frequency is periodic for drones, irregular for birds
4. âœ… Features can be extracted for classification

**Key takeaway:**
"Micro-Doppler is the signature that makes drone detection possible. Without it, drones look like birds in an RDM. The periodic blade flash is the fingerprint."

---

### PART 4: CFAR DETECTION

**Learning Objectives:**

1. Understand why fixed thresholds fail
2. Implement CA-CFAR, OS-CFAR algorithms
3. Compare performance in different clutter environments
4. Introduce learned detection networks (AI replacement)

**Prerequisites:** Parts 1-3, understand RDM and noise statistics

**Estimated Time:** 1-2 days

#### Section 1: The Detection Problem

**Concept to teach:**

- RDM has targets buried in noise/clutter
- Need automatic detection: "Is there a target in this cell?"
- Fixed threshold fails: too high (miss targets) or too low (false alarms)

**Physical intuition:**

```
Problem: Noise floor varies across RDM
- Near ground clutter: High noise floor
- Clear sky region: Low noise floor

Fixed threshold = 30 dB â†’ Works in sky, fails in clutter
Solution: ADAPTIVE threshold based on local statistics
```

**Visualization needed:**

1. RDM with varying clutter (strong at low ranges, weak at high)
2. Fixed threshold overlay (misses targets in clutter, false alarms in clear regions)
3. Adaptive threshold following noise floor

#### Section 2: CA-CFAR (Cell-Averaging CFAR)

**Concept to teach:**
Estimate noise floor by averaging cells around test cell

**The algorithm:**

```
For each cell in RDM:
    1. Define guard cells (immediate neighbors - exclude)
    2. Define training cells (surrounding ring - use for statistics)
    3. Average power in training cells â†’ noise estimate
    4. Threshold = noise_estimate * scale_factor
    5. If test_cell > threshold â†’ DETECTION
```

**Code structure:**

```python
def ca_cfar_2d(rdm, guard_cells=2, training_cells=8, pfa=1e-6):
    """
    2D Cell-Averaging CFAR detector

    Parameters:
    - rdm: Range-Doppler Map [Doppler_bins x Range_bins]
    - guard_cells: Number of guard cells around test cell
    - training_cells: Number of training cells for noise estimation
    - pfa: Probability of false alarm (determines scale factor)

    Returns:
    - detections: Binary mask [same size as rdm]
    - threshold_map: Adaptive threshold at each cell
    """
    # Calculate scale factor from Pfa
    scale_factor = calculate_scale_factor(pfa, num_training_cells)

    # For each cell:
    detections = np.zeros_like(rdm, dtype=bool)
    threshold_map = np.zeros_like(rdm)

    for i in range(rdm.shape[0]):
        for j in range(rdm.shape[1]):
            # Extract training cells (excluding guard cells)
            training_region = get_training_cells(rdm, i, j, guard_cells, training_cells)

            # Estimate noise floor
            noise_estimate = np.mean(training_region)

            # Adaptive threshold
            threshold = noise_estimate * scale_factor
            threshold_map[i, j] = threshold

            # Detection test
            if rdm[i, j] > threshold:
                detections[i, j] = True

    return detections, threshold_map
```

**Visualization:**

1. RDM with CFAR detections overlaid
2. Threshold map showing adaptive threshold surface
3. ROC curve (detection probability vs false alarm rate)

#### Section 3: OS-CFAR (Ordered Statistics CFAR)

**Concept to teach:**
More robust in clutter edges - use median instead of mean

**Why better than CA-CFAR:**

```
CA-CFAR problem: If training cells include both clutter and noise,
mean is pulled up â†’ threshold too high â†’ miss targets

OS-CFAR solution: Sort training cells, take k-th order statistic (often median)
â†’ Robust to outliers and clutter edges
```

**Implementation:**

```python
def os_cfar_2d(rdm, guard_cells=2, training_cells=8, k_order=0.75, pfa=1e-6):
    """
    Ordered Statistics CFAR

    Parameters:
    - k_order: Which order statistic to use (0.75 = 75th percentile)
    """
    # Similar to CA-CFAR but:
    # noise_estimate = np.percentile(training_region, k_order*100)
```

#### Section 4: Learned Detection Networks (AI Replacement)

**Concept to teach:**
Instead of hand-designed CFAR, train CNN to detect targets

**Architecture:**

```python
class CFARNet(nn.Module):
    """
    Convolutional network for target detection

    Input: RDM patch [1 x H x W]
    Output: Detection probability [0-1]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*6*6, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*6*6)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
```

**Training approach:**

```python
# Generate synthetic RDMs with known target locations
train_data = generate_synthetic_rdms(num_samples=10000)

# Train network
model = CFARNet()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCELoss()

for epoch in range(100):
    for rdm_patch, label in train_data:
        pred = model(rdm_patch)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
```

**Comparison visualization:**

1. CA-CFAR detections
2. OS-CFAR detections
3. Neural network detections
4. Performance metrics (Pd vs Pfa curves)

#### Section 5: Practical Examples

**Scenario: Multi-target in varying clutter**

```python
radar = PulseDopplerRadar(num_pulses=128, samples_per_pulse=512)

# Strong clutter at close range
for r in [500, 1000, 1500]:
    radar.add_target(range_m=r, velocity_ms=0, rcs=5.0)

# Weak targets in clutter
radar.add_target(range_m=1200, velocity_ms=20, rcs=0.1)  # Drone
radar.add_target(range_m=800, velocity_ms=8, rcs=0.05)   # Bird

# Targets in clear region
radar.add_target(range_m=5000, velocity_ms=-30, rcs=0.2)

radar.add_noise(snr_db=10)

rdm, range_axis, velocity_axis = radar.generate_rdm()

# Apply different detectors
det_ca, thresh_ca = ca_cfar_2d(rdm)
det_os, thresh_os = os_cfar_2d(rdm)
det_nn = neural_cfar(rdm, model)

# Compare results
plot_cfar_comparison(rdm, det_ca, det_os, det_nn)
```

#### Exercises for Part 4

1. **Pfa tuning:** Vary probability of false alarm from 1e-3 to 1e-9, see impact
2. **Guard/training cells:** Experiment with different window sizes
3. **Clutter edges:** Create scenario with sharp clutter edge, compare CA vs OS
4. **Neural network:** Train on different clutter statistics, test robustness

#### Reality Check for Part 4

**What you should now understand:**

1. âœ… Fixed thresholds fail in non-uniform noise
2. âœ… CFAR adapts threshold to local statistics
3. âœ… OS-CFAR more robust than CA-CFAR at clutter edges
4. âœ… Neural networks can learn detection but need training data

**Key takeaway:**
"CFAR solves the 'where are the targets?' problem automatically. The threshold adapts to the noise floor, maintaining constant false alarm rate."

---

### PART 5: CLUTTER SUPPRESSION

**Learning Objectives:**

1. Understand different clutter types (ground, weather, sea)
2. Implement classical filters (MTI, adaptive MTI)
3. Introduce STAP (Space-Time Adaptive Processing) conceptually
4. Implement neural network for clutter cancellation (AI replacement)

**Prerequisites:** Parts 1-4, understand Doppler processing and spectral filtering

**Estimated Time:** 2-3 days

#### Section 1: Clutter Characteristics

**Concept to teach:**
Different clutter types have different Doppler characteristics

**Clutter types:**

```
1. Ground clutter:
   - Stationary (v â‰ˆ 0)
   - Very strong (40-60 dB above targets)
   - Narrow Doppler spread

2. Weather clutter (rain, snow):
   - Low velocity (wind-driven)
   - Moderate strength
   - Broader Doppler spread

3. Sea clutter:
   - Moving with waves
   - Doppler spread from wave motion
   - Non-Gaussian statistics

4. Vegetation (wind-blown trees):
   - Time-varying velocity
   - Fluctuating strength
```

**Visualization needed:**

1. RDM showing different clutter types
2. Doppler spectra comparison (ground vs weather vs sea)
3. Clutter-to-noise ratio map

#### Section 2: Classical MTI Filters

**Review from Part 2, then extend:**

**Single-delay MTI (already covered):**

```python
# y[n] = x[n] - x[n-1]
# Removes DC, attenuates low velocities
```

**Three-pulse MTI (better clutter rejection):**

```python
def three_pulse_mti(data_matrix):
    """
    Three-pulse MTI filter

    Transfer function: H(f) = (1 - exp(-j*2Ï€*f))^2
    Better notch at zero Doppler than single-delay
    """
    filtered = np.zeros_like(data_matrix)
    filtered[2:, :] = data_matrix[2:, :] - 2*data_matrix[1:-1, :] + data_matrix[:-2, :]
    return filtered
```

**Adaptive MTI (adjusts to clutter characteristics):**

```python
def adaptive_mti(data_matrix, clutter_estimate):
    """
    Adjusts filter coefficients based on estimated clutter statistics

    Uses Wiener filter approach:
    - Estimate clutter covariance from data
    - Design filter to minimize clutter while preserving signal
    """
    # Estimate clutter subspace
    # Project out clutter
    # Return filtered data
```

#### Section 3: STAP Concept (Brief Overview)

**Concept to teach:**
Combine spatial (antenna array) and temporal (pulse-to-pulse) processing

**Why STAP:**

```
MTI: Only uses temporal information (across pulses)
â†’ Can reject clutter at specific velocities

STAP: Uses both space and time
â†’ Can reject clutter at specific velocities AND angles
â†’ Much better clutter rejection, especially for moving platforms
```

**Note:** Full STAP requires array radar. Just introduce concept:

```python
# Conceptual STAP
# For array radar with N elements, M pulses:
# Data cube: [N elements Ã— M pulses Ã— K range bins]
# STAP filter: Jointly process space-time to maximize SINR
```

**Focus on insight, not implementation** (too complex for tutorial scope)

#### Section 4: Neural Network for Clutter Cancellation (AI Replacement)

**Concept to teach:**
Train network to learn clutter characteristics and suppress adaptively

**Architecture:**

```python
class ClutterNet(nn.Module):
    """
    U-Net style architecture for clutter suppression

    Input: Raw RDM (with clutter) [1 x H x W]
    Output: Clutter-suppressed RDM [1 x H x W]
    """
    def __init__(self):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Decoder (upsampling)
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.dec3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encode
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(F.max_pool2d(e1, 2)))
        e3 = F.relu(self.enc3(F.max_pool2d(e2, 2)))

        # Decode
        d1 = F.relu(self.dec1(e3))
        d2 = F.relu(self.dec2(d1))
        out = self.dec3(d2)

        return out
```

**Training strategy:**

```python
# Generate pairs: (RDM with clutter, RDM without clutter)
def generate_training_pairs(num_samples=5000):
    pairs = []
    for _ in range(num_samples):
        # Create clean RDM (targets only)
        rdm_clean = generate_clean_rdm()

        # Add synthetic clutter
        rdm_cluttered = rdm_clean + generate_clutter()

        pairs.append((rdm_cluttered, rdm_clean))

    return pairs

# Train to reconstruct clean RDM from cluttered input
model = ClutterNet()
loss_fn = nn.MSELoss()

for epoch in range(200):
    for rdm_in, rdm_target in train_pairs:
        rdm_pred = model(rdm_in)
        loss = loss_fn(rdm_pred, rdm_target)
        # Backprop and optimize
```

**Advantages of neural approach:**

- Learns complex clutter patterns from data
- Adapts to different clutter environments
- Can handle non-stationary clutter
- Doesn't require explicit clutter model

**Disadvantages:**

- Requires training data
- Black box (hard to interpret)
- May not generalize to unseen clutter types

#### Section 5: Comparison Examples

**Scenario 1: Ground clutter only**

```python
# Strong stationary clutter
# Compare: No filter vs MTI vs 3-pulse MTI vs Neural
# Show MTI works well (clutter is stationary)
```

**Scenario 2: Weather clutter**

```python
# Moderate, spread clutter
# Compare: MTI vs Adaptive MTI vs Neural
# Show adaptive methods work better
```

**Scenario 3: Mixed clutter**

```python
# Ground + weather + sea
# Show neural network adapts best
```

#### Exercises for Part 5

1. **MTI depth:** Compare 1-pulse, 2-pulse, 3-pulse MTI - how does notch depth change?
2. **Clutter velocity:** Add moving clutter (e.g., cars at 20 m/s) - can MTI handle it?
3. **Neural training:** Train on one clutter type, test on another - does it generalize?
4. **SINR curves:** Plot Signal-to-Interference-plus-Noise Ratio vs filter type

#### Reality Check for Part 5

**What you should now understand:**

1. âœ… Clutter characteristics vary (stationary, moving, spread)
2. âœ… MTI filters remove DC (zero Doppler)
3. âœ… Adaptive filters adjust to clutter statistics
4. âœ… Neural networks can learn to suppress complex clutter patterns

**Key takeaway:**
"Clutter suppression is about removing unwanted returns so weak targets become visible. Classical MTI works for stationary clutter, but adaptive/neural methods handle complex environments."

---

### PART 6: CLASSIFICATION (Drone vs Bird vs Aircraft)

**Learning Objectives:**

1. Extract discriminative features from RDMs and spectrograms
2. Implement classical ML classifier (SVM, Random Forest)
3. Implement CNN for RDM classification
4. Implement Transformer for spectrogram classification
5. Compare performance and discuss trade-offs

**Prerequisites:** Parts 1-5, especially micro-Doppler (Part 3)

**Estimated Time:** 3-4 days

#### Section 1: The Classification Problem

**Concept to teach:**
Given a detection, determine what type of object it is

**Classes:**

```
1. Drone (quadcopter)
   - Micro-Doppler: Periodic blade flash, high RPM
   - RCS: Small (0.001 - 0.1 mÂ²)
   - Velocity: 5-25 m/s

2. Bird
   - Micro-Doppler: Irregular wing flutter, low frequency
   - RCS: Very small (0.0001 - 0.01 mÂ²)
   - Velocity: 5-20 m/s (similar to drone!)

3. Aircraft
   - Micro-Doppler: Propeller (if present), much larger blade
   - RCS: Large (1 - 100 mÂ²)
   - Velocity: 50-200 m/s

4. Ground vehicle (car, truck)
   - Micro-Doppler: Wheel rotation (if visible)
   - RCS: Moderate (0.1 - 10 mÂ²)
   - Velocity: 10-40 m/s, but constrained to roads
```

**Key challenge:** Drone vs Bird separation (similar size, similar velocity)
**Solution:** Micro-Doppler signatures!

**Visualization:**

1. Example spectrograms of each class side-by-side
2. Feature space scatter plot (e.g., RCS vs velocity vs blade flash frequency)

#### Section 2: Feature-Based Classification (Classical ML)

**Features to extract:**

```python
def extract_classification_features(rdm, spectrogram, range_bin, velocity_bin):
    """
    Extract comprehensive feature set

    Returns:
    - features: Dictionary of features for ML classifier
    """
    features = {}

    # 1. Kinematic features (from RDM)
    features['range'] = range_bin * range_resolution
    features['velocity'] = velocity_bin * velocity_resolution
    features['rcs_estimate'] = rdm[velocity_bin, range_bin]

    # 2. Micro-Doppler features (from spectrogram)
    features['blade_flash_freq'] = extract_bff(spectrogram)
    features['modulation_depth'] = np.ptp(spectrogram, axis=1).mean()
    features['periodicity'] = measure_periodicity(spectrogram)
    features['bandwidth'] = spectrogram.std()

    # 3. Statistical features
    features['spectral_entropy'] = compute_entropy(spectrogram)
    features['centroid_variance'] = np.var(compute_centroid(spectrogram))

    return features
```

**Classical ML approach:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Generate training data
X_train, y_train = generate_labeled_dataset(num_samples=5000)
# X_train: [num_samples x num_features]
# y_train: [num_samples] with labels {0: drone, 1: bird, 2: aircraft, 3: vehicle}

# Train classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(X_train, y_train)

# Test
X_test, y_test = generate_test_dataset(num_samples=1000)
accuracy = clf.score(X_test, y_test)

# Feature importance
importance = clf.feature_importances_
# Plot which features matter most
```

**Visualization:**

1. Feature importance bar chart
2. Confusion matrix
3. Decision boundary plots (2D projections)

#### Section 3: CNN for RDM Classification

**Concept to teach:**
Instead of hand-crafted features, let CNN learn features directly from RDM

**Architecture:**

```python
class RDMClassifier(nn.Module):
    """
    CNN for classifying RDM patches

    Input: RDM patch [1 x 64 x 64] (cropped around detection)
    Output: Class probabilities [4] (drone, bird, aircraft, vehicle)
    """
    def __init__(self, num_classes=4):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # 64->32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # 32->16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)  # 16->8

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.softmax(x, dim=1)
```

**Training approach:**

```python
# Generate synthetic RDM dataset
train_dataset = generate_rdm_dataset(
    num_per_class=2000,
    classes=['drone', 'bird', 'aircraft', 'vehicle']
)

# Data augmentation
# - Add noise at different SNRs
# - Vary RCS
# - Vary range/velocity within patch

model = RDMClassifier(num_classes=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    for rdm_patch, label in train_loader:
        pred = model(rdm_patch)
        loss = loss_fn(pred, label)
        # Backprop
```

**Visualization:**

1. Training curves (loss, accuracy)
2. Confusion matrix on test set
3. Grad-CAM visualization (what parts of RDM the network focuses on)

#### Section 4: Transformer for Spectrogram Classification

**Concept to teach:**
Spectrograms are like images, but also have sequential structure (time evolution)
Transformers can capture long-range dependencies in micro-Doppler modulation

**Architecture:**

```python
class SpectrogramTransformer(nn.Module):
    """
    Vision Transformer for spectrogram classification

    Input: Spectrogram [1 x T x F] (time x frequency)
    Output: Class probabilities [4]
    """
    def __init__(self, num_classes=4, patch_size=8, dim=256, depth=6, heads=8):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, (T//patch_size)*(F//patch_size), dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        return F.softmax(self.mlp_head(x), dim=1)
```

**Why Transformer for spectrograms:**

- Captures temporal evolution of blade flash
- Attention mechanism can focus on periodic modulation
- Better than CNN for long-range dependencies

#### Section 5: Comparison and Ensemble

**Create comprehensive comparison:**

```python
# Test all methods on same dataset
results = {
    'Random Forest': test_random_forest(X_test, y_test),
    'SVM': test_svm(X_test, y_test),
    'CNN': test_cnn(rdm_patches_test, y_test),
    'Transformer': test_transformer(spectrogram_test, y_test)
}

# Plot comparison
plot_accuracy_comparison(results)
plot_confusion_matrices(results)
plot_roc_curves(results)
```

**Ensemble classifier:**

```python
def ensemble_classifier(rdm_patch, spectrogram):
    """
    Combine predictions from multiple models
    """
    # CNN on RDM
    pred_cnn = cnn_model(rdm_patch)

    # Transformer on spectrogram
    pred_transformer = transformer_model(spectrogram)

    # Random Forest on features
    features = extract_features(rdm_patch, spectrogram)
    pred_rf = rf_model.predict_proba(features)

    # Weighted average
    ensemble_pred = 0.4*pred_cnn + 0.4*pred_transformer + 0.2*pred_rf

    return np.argmax(ensemble_pred)
```

#### Section 6: Real-World Considerations

**Discuss practical issues:**

1. **Class imbalance:** More birds than drones in real data
2. **Unknown classes:** What if it's not drone/bird/aircraft?
3. **Confidence thresholding:** When to say "I don't know"
4. **Adversarial examples:** Can targets spoof classification?

**Code example:**

```python
# Confidence-based rejection
def classify_with_confidence(model, input_data, threshold=0.8):
    probs = model(input_data)
    max_prob = np.max(probs)

    if max_prob < threshold:
        return "UNKNOWN", max_prob
    else:
        return class_names[np.argmax(probs)], max_prob
```

#### Exercises for Part 6

1. **Feature engineering:** Add new features, see if accuracy improves
2. **Architecture search:** Try different CNN depths, widths
3. **Data augmentation:** Vary SNR, clutter during training - does it help?
4. **Transfer learning:** Train on simulated data, test on real data (if available)

#### Reality Check for Part 6

**What you should now understand:**

1. âœ… Micro-Doppler features separate drone from bird
2. âœ… Classical ML needs hand-crafted features
3. âœ… CNNs learn features automatically from RDM
4. âœ… Transformers capture temporal patterns in spectrograms
5. âœ… Ensemble methods combine strengths of different approaches

**Key takeaway:**
"Classification is the intelligence of the radar system. Micro-Doppler signatures make drone detection possible, and deep learning extracts those signatures automatically."

---

### PART 7: TRACKING (OPTIONAL - Time Permitting)

**Learning Objectives:**

1. Understand data association problem
2. Implement Kalman filter for single target tracking
3. Introduce multi-target tracking (JPDA, MHT concepts)
4. Show LSTM/Transformer for track prediction (AI replacement)

**Prerequisites:** Parts 1-6

**Estimated Time:** 2-3 days (if time allows)

**Note:** This is lower priority. If time is tight, skip or provide minimal treatment.

#### Section 1: The Tracking Problem

**Concept:**

- Detection gives you snapshots: "target at (R, V) at time t"
- Tracking connects snapshots: "target is following this trajectory"

**Why needed:**

- Smooth noisy detections
- Predict future position
- Maintain target identity (Track ID)

**Challenges:**

- Missed detections (target temporarily invisible)
- False alarms (clutter looks like target)
- Data association (which detection belongs to which track?)

#### Section 2: Kalman Filter (Classical)

**State space model:**

```
State: x = [range, velocity, acceleration]
Measurement: z = [range, velocity] (from RDM detection)

Predict: x_pred = F * x_prev
Update: x_new = x_pred + K * (z - H*x_pred)
```

**Implementation:**

```python
class KalmanTracker:
    def __init__(self, initial_state):
        self.x = initial_state  # [r, v, a]
        self.P = np.eye(3) * 10  # Covariance
        self.F = np.array([[1, dt, 0.5*dt**2],
                          [0, 1, dt],
                          [0, 0, 1]])
        self.H = np.array([[1, 0, 0],
                          [0, 1, 0]])
        self.Q = process_noise  # Process noise
        self.R = measurement_noise  # Measurement noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P
```

#### Section 3: Multi-Target Tracking Concept

**Brief overview only (full implementation too complex):**

- JPDA: Joint Probabilistic Data Association
- MHT: Multiple Hypothesis Tracking

**Focus on the problem:**

```
At time t: 3 tracks, 4 detections
Which detection goes with which track?
- Nearest neighbor (simple but fails)
- Global nearest neighbor (better)
- Probabilistic assignment (JPDA)
```

#### Section 4: LSTM for Track Prediction (AI Replacement)

**Concept:**
Learn motion patterns from data instead of assuming constant velocity/acceleration

**Architecture:**

```python
class TrackPredictor(nn.Module):
    """
    LSTM to predict next position given track history

    Input: Track history [seq_len x 2] (range, velocity)
    Output: Next position [2]
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, num_layers=2)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, track_history):
        # track_history: [seq_len, batch, 2]
        out, (h, c) = self.lstm(track_history)
        pred = self.fc(out[-1])  # Last output
        return pred
```

**Advantages over Kalman:**

- Learns non-linear motion patterns
- Can handle maneuvering targets
- Adapts to specific target behaviors

**Keep this section brief** - just show the concept, not full implementation.

#### Reality Check for Part 7

**What you should now understand:**

1. âœ… Tracking connects detections over time
2. âœ… Kalman filter assumes linear motion
3. âœ… Data association is the hard problem
4. âœ… LSTMs can learn complex motion patterns

**Key takeaway:**
"Tracking turns snapshots into trajectories. Kalman filter works for simple motion, but learning-based methods handle maneuvering targets better."

---

### PART 8: COGNITIVE RADAR FRAMEWORK

**Learning Objectives:**

1. Understand agent-based architecture for radar processing
2. Design decision-making agents for each processing stage
3. Implement simple reinforcement learning for parameter optimization
4. Build end-to-end cognitive radar system

**Prerequisites:** Parts 1-6 (Part 7 optional)

**Estimated Time:** 2-3 days

**THIS IS THE CAPSTONE - Ties everything together**

#### Section 1: What is Cognitive Radar?

**Concept to teach:**
Radar that adapts itself to environment and mission

**Traditional radar:**

```
Fixed parameters â†’ Process data â†’ Output detections
```

**Cognitive radar:**

```
Sense environment â†’ Decide parameters â†’ Process data â†’ Learn from results â†’ Repeat
```

**Key capabilities:**

1. **Perception:** Understand current environment (clutter type, interference, etc.)
2. **Decision:** Choose processing parameters (filter type, threshold, waveform)
3. **Learning:** Improve decisions based on past performance
4. **Adaptation:** Change behavior as environment changes

**Analogy:**
"Like a skilled radar operator who adjusts knobs based on what they see. The cognitive radar automates this expert knowledge."

#### Section 2: Agent-Based Architecture

**Design multi-agent system:**

```python
class RadarAgent:
    """
    Base class for radar processing agents
    """
    def __init__(self, name):
        self.name = name
        self.state = {}
        self.action_space = []
        self.reward_history = []

    def perceive(self, environment):
        """Observe current environment state"""
        raise NotImplementedError

    def decide(self):
        """Choose action based on state"""
        raise NotImplementedError

    def act(self, data):
        """Execute action on data"""
        raise NotImplementedError

    def learn(self, reward):
        """Update policy based on reward"""
        raise NotImplementedError
```

**Specific agents:**

```python
class ClutterSuppressionAgent(RadarAgent):
    """
    Decides which clutter filter to use

    Actions:
    - No filter (clear environment)
    - MTI (stationary clutter)
    - 3-pulse MTI (strong stationary clutter)
    - Adaptive filter (non-stationary clutter)
    - Neural network (complex clutter)
    """
    def __init__(self):
        super().__init__("ClutterSuppression")
        self.action_space = ['none', 'mti', '3pulse', 'adaptive', 'neural']
        self.policy_net = self._build_policy_network()

    def perceive(self, rdm):
        """
        Extract environment features:
        - Clutter-to-noise ratio
        - Doppler spread at v=0
        - Temporal variation
        """
        features = {
            'cnr': estimate_cnr(rdm),
            'doppler_spread': measure_spread(rdm, velocity=0),
            'temporal_var': estimate_variation(rdm)
        }
        self.state = features
        return features

    def decide(self):
        """
        Use policy network to choose filter
        """
        state_vector = self._encode_state(self.state)
        action_probs = self.policy_net(state_vector)
        action = self.action_space[torch.argmax(action_probs)]
        return action

    def act(self, data_matrix):
        """
        Apply chosen filter to data
        """
        action = self.decide()

        if action == 'none':
            return data_matrix
        elif action == 'mti':
            return apply_mti(data_matrix)
        elif action == '3pulse':
            return apply_3pulse_mti(data_matrix)
        # ... etc

    def learn(self, reward):
        """
        Update policy based on reward (e.g., SINR improvement)
        """
        # Reinforcement learning update
        # Store experience, update network
```

**Other agents:**

```python
class DetectionAgent(RadarAgent):
    """
    Decides detection threshold and algorithm

    Actions:
    - CA-CFAR with Pfa âˆˆ {1e-3, 1e-4, 1e-5, 1e-6}
    - OS-CFAR with k âˆˆ {0.6, 0.7, 0.8, 0.9}
    - Neural detector
    """

class ClassificationAgent(RadarAgent):
    """
    Decides which classifier to use

    Actions:
    - Quick feature-based (low compute)
    - CNN on RDM (medium compute)
    - Transformer on spectrogram (high compute, high accuracy)
    - Ensemble (highest accuracy, most compute)
    """

class WaveformAgent(RadarAgent):
    """
    Decides radar waveform parameters (advanced)

    Actions:
    - PRF selection
    - Bandwidth selection
    - Pulse width
    - Number of pulses in CPI
    """
```

#### Section 3: Communication Between Agents

**Agents share information:**

```python
class CognitiveRadarSystem:
    """
    Orchestrates multiple agents
    """
    def __init__(self):
        self.agents = {
            'waveform': WaveformAgent(),
            'clutter': ClutterSuppressionAgent(),
            'detection': DetectionAgent(),
            'classification': ClassificationAgent(),
            'tracking': TrackingAgent()
        }
        self.blackboard = {}  # Shared memory

    def process_cpi(self, raw_data):
        """
        Process one coherent processing interval
        """
        # Stage 1: Waveform agent observes environment
        self.blackboard['interference'] = self.agents['waveform'].perceive(raw_data)

        # Stage 2: Generate RDM
        rdm = generate_rdm(raw_data)

        # Stage 3: Clutter suppression
        rdm_clean = self.agents['clutter'].act(rdm)
        self.blackboard['rdm'] = rdm_clean

        # Stage 4: Detection
        detections = self.agents['detection'].act(rdm_clean)
        self.blackboard['detections'] = detections

        # Stage 5: Classification
        classifications = []
        for det in detections:
            cls = self.agents['classification'].act(det)
            classifications.append(cls)

        # Stage 6: Tracking
        tracks = self.agents['tracking'].act(detections, classifications)

        # Calculate reward for each agent
        self._compute_rewards()

        # Agents learn
        for agent in self.agents.values():
            agent.learn(self.blackboard['rewards'][agent.name])

        return tracks

    def _compute_rewards(self):
        """
        Compute reward signal for each agent

        Examples:
        - Clutter agent: SINR improvement
        - Detection agent: True positive rate (if ground truth available)
        - Classification agent: Accuracy
        - Tracking agent: Track continuity
        """
        rewards = {}

        # Clutter suppression reward
        sinr_before = compute_sinr(self.blackboard['rdm_raw'])
        sinr_after = compute_sinr(self.blackboard['rdm'])
        rewards['clutter'] = sinr_after - sinr_before

        # Detection reward (if we have ground truth)
        if 'ground_truth' in self.blackboard:
            tp = count_true_positives(self.blackboard['detections'],
                                     self.blackboard['ground_truth'])
            fp = count_false_positives(self.blackboard['detections'],
                                      self.blackboard['ground_truth'])
            rewards['detection'] = tp - 0.1*fp  # Penalize false alarms

        # ... etc for other agents

        self.blackboard['rewards'] = rewards
```

#### Section 4: Reinforcement Learning for Parameter Optimization

**Simple Q-learning example:**

```python
class QLearningAgent(RadarAgent):
    """
    Agent using Q-learning for discrete action space
    """
    def __init__(self, state_dim, action_space):
        super().__init__("QLearning")
        self.action_space = action_space
        self.Q = np.zeros((state_dim, len(action_space)))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def decide(self):
        """
        Epsilon-greedy action selection
        """
        state_idx = self._discretize_state(self.state)

        if np.random.rand() < self.epsilon:
            # Explore
            action_idx = np.random.randint(len(self.action_space))
        else:
            # Exploit
            action_idx = np.argmax(self.Q[state_idx, :])

        return self.action_space[action_idx]

    def learn(self, reward):
        """
        Q-learning update
        """
        s = self._discretize_state(self.state)
        a = self.action_space.index(self.last_action)
        s_next = self._discretize_state(self.state_next)

        # Q-learning update rule
        self.Q[s, a] += self.alpha * (
            reward + self.gamma * np.max(self.Q[s_next, :]) - self.Q[s, a]
        )
```

**Deep Q-Network (DQN) for continuous state:**

```python
class DQNAgent(RadarAgent):
    """
    Agent using Deep Q-Network
    """
    def __init__(self, state_dim, action_space):
        super().__init__("DQN")
        self.action_space = action_space
        self.q_network = self._build_q_network(state_dim, len(action_space))
        self.target_network = self._build_q_network(state_dim, len(action_space))
        self.replay_buffer = []
        self.optimizer = torch.optim.Adam(self.q_network.parameters())

    def _build_q_network(self, state_dim, num_actions):
        """
        Neural network to approximate Q-values
        """
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def decide(self):
        """
        Choose action using Q-network
        """
        state_tensor = torch.FloatTensor(self._encode_state(self.state))
        q_values = self.q_network(state_tensor)

        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(len(self.action_space))
        else:
            action_idx = torch.argmax(q_values).item()

        return self.action_space[action_idx]

    def learn(self, reward):
        """
        Experience replay and network update
        """
        # Store experience
        self.replay_buffer.append((self.state, self.last_action,
                                  reward, self.state_next))

        # Sample minibatch
        if len(self.replay_buffer) > 32:
            batch = random.sample(self.replay_buffer, 32)

            # Compute loss and update
            loss = self._compute_dqn_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Periodic target network update
        if self.update_counter % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
```

#### Section 5: End-to-End Example

**Scenario: Cognitive radar adapting to changing environment**

```python
# Initialize system
cognitive_radar = CognitiveRadarSystem()

# Simulate changing environment
environments = [
    {'type': 'clear', 'duration': 100},      # Clear sky
    {'type': 'ground_clutter', 'duration': 100},  # Strong ground returns
    {'type': 'weather', 'duration': 100},    # Rain
    {'type': 'mixed', 'duration': 100}       # Ground + weather
]

# Run simulation
for env in environments:
    print(f"\n--- Environment: {env['type']} ---")

    for cpi in range(env['duration']):
        # Generate data for this environment
        raw_data = generate_radar_data(env['type'])

        # Process through cognitive system
        tracks = cognitive_radar.process_cpi(raw_data)

        # Log decisions
        if cpi % 10 == 0:
            print(f"CPI {cpi}:")
            print(f"  Clutter filter: {cognitive_radar.agents['clutter'].last_action}")
            print(f"  Detection Pfa: {cognitive_radar.agents['detection'].last_action}")
            print(f"  Avg reward: {np.mean(list(cognitive_radar.blackboard['rewards'].values()))}")

# Plot learning curves
plot_agent_learning_curves(cognitive_radar)
```

**Visualization:**

1. Agent decisions over time (which filter, which threshold)
2. Performance metrics vs time (SINR, detection rate, classification accuracy)
3. Reward curves showing agents improving
4. Comparison: Cognitive vs fixed-parameter system

#### Section 6: Practical Considerations

**Discuss real-world deployment:**

1. **Computational constraints**

   - Neural networks take time to compute
   - Agent must decide quickly (real-time requirement)
   - Trade-off: Performance vs latency

2. **Safety and fallbacks**

   - What if agent makes bad decision?
   - Fallback to classical methods
   - Human override capability

3. **Transfer learning**

   - Train in simulation, deploy in real world
   - Domain adaptation techniques

4. **Explainability**
   - Why did agent choose this action?
   - Logging and visualization of decision process

**Code example:**

```python
class ExplainableAgent(RadarAgent):
    """
    Agent that logs its reasoning
    """
    def decide(self):
        action = super().decide()

        # Log explanation
        self.explanation = {
            'state': self.state,
            'action': action,
            'q_values': self.q_network(self.state_tensor).detach().numpy(),
            'reasoning': self._generate_explanation()
        }

        return action

    def _generate_explanation(self):
        """
        Human-readable explanation of decision
        """
        if self.state['cnr'] > 40:
            return "High clutter detected, choosing strong MTI filter"
        elif self.state['doppler_spread'] > 5:
            return "Spread clutter, using adaptive filter"
        else:
            return "Low clutter, minimal filtering to preserve SNR"
```

#### Exercises for Part 8

1. **Add new agent:** Create a "Waveform Agent" that selects PRF based on environment
2. **Reward shaping:** Experiment with different reward functions - which works best?
3. **Multi-objective:** Agents optimize for both accuracy AND computational cost
4. **Adversarial environment:** Add jamming or deception - can agents adapt?

#### Reality Check for Part 8

**What you should now understand:**

1. âœ… Cognitive radar = self-adapting system
2. âœ… Each processing stage can be an autonomous agent
3. âœ… Agents learn optimal policies through RL
4. âœ… Multi-agent systems require coordination and communication
5. âœ… Real-world deployment requires safety, explainability, efficiency

**Key takeaway:**
"Cognitive radar brings AI to every stage of processing. Instead of fixed algorithms, intelligent agents adapt to the environment, learn from experience, and optimize performance dynamically. This is the future of radar systems."

---

## 7. CODE QUALITY STANDARDS

### General Principles

```python
# GOOD: Clear, documented, production-ready
def compute_range_resolution(bandwidth_hz):
    """
    Calculate minimum resolvable range difference

    Parameters:
    - bandwidth_hz: Signal bandwidth [Hz]

    Returns:
    - resolution: Range resolution [m]

    Physics: Resolution = c / (2*B)
    where c = speed of light, B = bandwidth
    """
    c = 3e8  # Speed of light [m/s]
    if bandwidth_hz <= 0:
        raise ValueError("Bandwidth must be positive")

    resolution = c / (2 * bandwidth_hz)
    return resolution

# BAD: No docs, no validation, unclear names
def calc(b):
    return 3e8/(2*b)
```

### Class Design

- Inherit from existing classes when appropriate (PulseDopplerRadar â†’ MicroDopplerRadar)
- Use composition for complex functionality
- Keep methods focused (single responsibility)
- Properties for derived quantities

### Error Handling

```python
# Always validate inputs
def add_target(self, range_m, velocity_ms, rcs=1.0):
    if range_m < 0:
        raise ValueError(f"Range must be non-negative, got {range_m}")
    if rcs <= 0:
        raise ValueError(f"RCS must be positive, got {rcs}")

    # ... implementation
```

### No Placeholder Code

```python
# NEVER do this:
def advanced_feature():
    # TODO: Implement later
    pass

# NEVER do this:
clutter = np.random.randn(100)  # Placeholder data

# If feature isn't ready, don't include it
# If you need synthetic data, make it realistic and explain why
```

---

## 8. VISUALIZATION GUIDELINES

### Standard Plot Structure

```python
def plot_standard_example():
    """
    Template for good visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time domain
    axes[0, 0].plot(time, signal, linewidth=2, color='blue')
    axes[0, 0].set_xlabel('Time [ms]', fontsize=11)
    axes[0, 0].set_ylabel('Amplitude', fontsize=11)
    axes[0, 0].set_title('Time Domain Signal', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, linestyle=':')

    # Plot 2: Frequency domain
    axes[0, 1].plot(freq, spectrum, linewidth=2, color='red')
    axes[0, 1].set_xlabel('Frequency [Hz]', fontsize=11)
    axes[0, 1].set_ylabel('Magnitude [dB]', fontsize=11)
    axes[0, 1].set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, linestyle=':')

    # Plot 3: 2D visualization (RDM, spectrogram)
    im = axes[1, 0].imshow(
        data_2d,
        aspect='auto',
        extent=[x_min, x_max, y_min, y_max],
        cmap='jet',
        origin='lower',
        vmin=vmin,
        vmax=vmax
    )
    axes[1, 0].set_xlabel('Range [km]', fontsize=11)
    axes[1, 0].set_ylabel('Velocity [m/s]', fontsize=11)
    axes[1, 0].set_title('Range-Doppler Map', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 0], label='Power [dB]')

    # Plot 4: Comparison or statistics
    axes[1, 1].bar(categories, values, color='green', alpha=0.7)
    axes[1, 1].set_xlabel('Category', fontsize=11)
    axes[1, 1].set_ylabel('Performance', fontsize=11)
    axes[1, 1].set_title('Performance Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    # plt.savefig('figure_name.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Color Scheme Rules

- **RDMs:** Always use 'jet' (industry standard, familiar to radar engineers)
- **Spectrograms:** 'viridis', 'magma', or 'inferno' (perceptually uniform)
- **Time series:** Use distinct colors from a qualitative palette
- **Heatmaps:** Diverging colormaps ('RdBu', 'seismic') if data has meaningful zero

### Always Include

- Axis labels with units in brackets [Hz], [m], [dB]
- Title describing what's shown
- Colorbar for 2D plots with label
- Grid for easier reading (alpha=0.3 for subtlety)
- Legend when multiple series
- Reference lines (e.g., v=0 line in RDM)

---

## 9. SIZE MANAGEMENT STRATEGY

### Monitoring

```python
# Include at end of each notebook
import os
notebook_path = 'notebook_name.ipynb'
size_bytes = os.path.getsize(notebook_path)
size_mb = size_bytes / 1e6

print(f"Notebook size: {size_mb:.2f} MB")
if size_mb > 0.9:
    print("âš ï¸  WARNING: Approaching 1 MB limit!")
if size_mb > 1.0:
    print("âŒ ERROR: Exceeds 1 MB limit - must reduce size")
```

### Reduction Strategies

**1. Limit output cell data**

```python
# Instead of printing huge arrays:
print(large_array)  # BAD

# Do this:
print(f"Array shape: {large_array.shape}")
print(f"Min: {large_array.min():.2f}, Max: {large_array.max():.2f}, Mean: {large_array.mean():.2f}")
```

**2. Clear outputs before saving**

- In Jupyter: Cell â†’ All Output â†’ Clear
- Save notebook with cleared outputs periodically

**3. Don't embed images**

```python
# Save to file instead of inline
plt.savefig('output.png', dpi=150)
plt.close()  # Don't display inline
```

**4. Limit array sizes in examples**

```python
# Keep examples manageable
num_samples = 512  # Not 4096
num_pulses = 128   # Not 1024
```

**5. Split if necessary**
If Part approaches 1 MB:

- Split into Part 3a and Part 3b
- Part 3a: Concepts and basic examples
- Part 3b: Advanced examples and exercises

---

## 10. ASSESSMENT & VALIDATION CRITERIA

### For Each Part, Student Should Be Able To:

**Part 3 (Micro-Doppler):**

- [ ] Explain physically why rotor blades create frequency modulation
- [ ] Generate spectrogram from radar data
- [ ] Identify blade flash signature visually
- [ ] Extract blade flash frequency programmatically
- [ ] Distinguish drone spectrogram from bird spectrogram

**Part 4 (CFAR):**

- [ ] Explain why fixed threshold fails
- [ ] Implement CA-CFAR algorithm from scratch
- [ ] Explain difference between CA-CFAR and OS-CFAR
- [ ] Tune Pfa to balance detection vs false alarms
- [ ] Train simple neural network for detection

**Part 5 (Clutter Suppression):**

- [ ] Identify different clutter types from RDM
- [ ] Implement MTI filter
- [ ] Explain why MTI removes DC component
- [ ] Train neural network for clutter cancellation
- [ ] Compare classical vs neural clutter suppression

**Part 6 (Classification):**

- [ ] Extract micro-Doppler features for classification
- [ ] Train Random Forest classifier on features
- [ ] Implement CNN for RDM classification
- [ ] Explain why transformer works for spectrograms
- [ ] Build ensemble classifier combining multiple models

**Part 7 (Tracking - Optional):**

- [ ] Implement Kalman filter for single target
- [ ] Explain data association problem
- [ ] Use LSTM to predict next position

**Part 8 (Cognitive Radar):**

- [ ] Design agent for one processing stage
- [ ] Implement simple Q-learning agent
- [ ] Coordinate multiple agents in system
- [ ] Compute appropriate reward signal
- [ ] Compare cognitive vs fixed-parameter system

### Code Validation

Every code cell should:

- [ ] Run without errors
- [ ] Produce expected output
- [ ] Complete in reasonable time (<10 seconds for examples)
- [ ] Be self-contained (no external files unless specified)

### Conceptual Validation

Every major concept should have:

- [ ] Physical intuition explanation
- [ ] Mathematical formulation
- [ ] Working code example
- [ ] Visualization
- [ ] Reality check / self-assessment

---

## APPENDIX: QUICK REFERENCE

### Key Equations

**Doppler frequency:**

```
f_d = 2 * v_radial / Î»
```

**Range resolution:**

```
Î”R = c / (2 * B)
```

**Velocity resolution:**

```
Î”v = Î» * PRF / (2 * M)
```

**Max unambiguous range:**

```
R_max = c / (2 * PRF)
```

**Max unambiguous velocity:**

```
v_max = Î» * PRF / 4
```

**Micro-Doppler from rotor:**

```
f_micro(t) = (4Ï€ / Î») * r_blade * Ï‰ * sin(Ï‰*t)
```

**CFAR threshold:**

```
threshold = noise_estimate * scale_factor(Pfa)
```

### Standard Radar Parameters (Use for Examples)

```python
# Typical X-band counter-UAS radar
f_c = 10e9          # 10 GHz
wavelength = 0.03   # 3 cm
PRF = 10e3          # 10 kHz
PRI = 100e-6        # 100 Î¼s
bandwidth = 150e6   # 150 MHz
num_pulses = 128    # CPI length
samples_per_pulse = 512

# Range resolution: 1 m
# Velocity resolution: 1.17 m/s
# Max range: 15 km
# Max velocity: 75 m/s
```

### Common Target Parameters

```python
# Drone (quadcopter)
drone = {
    'range': 1500,      # m
    'velocity': 15,     # m/s
    'rcs': 0.01,        # mÂ²
    'rpm': 3000,        # Rotor speed
    'num_blades': 4,
    'blade_length': 0.15  # m
}

# Bird
bird = {
    'range': 800,
    'velocity': 8,
    'rcs': 0.005,
    'wing_beat': 8      # Hz (irregular)
}

# Aircraft
aircraft = {
    'range': 5000,
    'velocity': 50,
    'rcs': 1.0,
    'prop_rpm': 2400    # If propeller
}
```

---

## REVISION HISTORY

**Version 1.2 (Oct 12, 2025):**

- Added ipywidgets setup pattern with installation check
- Included complete widget helper function template
- Added usage examples for embedded questions

**Version 1.1 (Oct 12, 2025):**

- Added intuition-testing questions before formulas
- Updated teaching flow: Concept â†’ Intuition Check â†’ Formula â†’ Code â†’ Experiment
- Added detailed guidelines for creating effective intuition questions
- Updated markdown cell template to include testing section

**Version 1.0 (Oct 10, 2025):**

- Initial specification
- Complete curriculum outline Parts 1-8
- Teaching style guidelines
- Code quality standards
- Size management strategy

---

## USAGE INSTRUCTIONS

**To create tutorials from this specification:**

1. Read entire specification first
2. For each Part, follow the detailed section specifications
3. Maintain consistent style throughout (see Section 3)
4. Build on previous parts (use existing classes/functions)
5. Keep notebooks under 1 MB (see Section 9)
6. Validate against assessment criteria (Section 10)
7. Test all code cells before delivery
8. No company names in code/notebooks

**Delivery format:**

- Jupyter notebooks (.ipynb)
- Python files (.py) as backup
- Clear filenames: `part3_micro_doppler.ipynb`

**Quality checklist before delivery:**

- [ ] All code runs without errors
- [ ] Visualizations are clear and properly labeled
- [ ] Physical intuition before math in every section
- [ ] Reality checks included
- [ ] Exercises provided
- [ ] File size < 1 MB
- [ ] No company references
- [ ] Consistent with existing Parts 1-2

---

END OF SPECIFICATION
