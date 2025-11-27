# Z-Scan Analysis for Transparent Materials

Python software for extracting **nonlinear refractive index (n₂)** from Z-scan measurements of transparent materials like YAG, where two-photon absorption is negligible.

This implementation adapts the open-source framework by [Arango et al.](https://github.com/juanjosearango/z-scan-parameter-extraction-program) for transparent dielectrics where traditional open aperture analysis is unreliable.

**Modified by**: Isuru Withanawasam  
**Purpose**: Specialized analysis of transparent media with negligible nonlinear absorption

---

## Experimental Configuration

Pre-configured for Nd:YAG measurements at ELI-ALPS:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Sample** | YAG, 2 mm thick | Transparent at test wavelength |
| **Wavelength (λ₀)** | 1060 nm | Nd:YAG laser line |
| **Pulse Duration (τ)** | 90 fs | Nominal value |
| **Repetition Rate** | 100 MHz | Nominal (peak power manually specified) |
| **Focusing Lens (f)** | 150 mm | Determines beam waist |
| **Aperture Radius (a)** | 0.75 mm | Closed aperture diameter: 1.5 mm |
| **Linear Absorption (α)** | 0.0 | Negligible for YAG at 1060 nm |
| **Nonlinear Absorption (β)** | 0.0 | TPA bypassed for transparent media |
| **Setup Length (L)** | 300 mm | Lens-to-aperture distance (far-field) |

---

## Analysis Method

### Beam Waist Determination

Traditional Z-scan uses the open aperture dip to find the beam waist. For transparent materials without nonlinear absorption, this code uses **geometric extraction from the closed aperture trace**:

1. Identify peak position (z_peak) and valley position (z_valley) in the normalized transmission
2. Calculate the peak-valley separation: ΔZ_pv = |z_peak - z_valley|
3. Estimate Rayleigh range: z₀ ≈ ΔZ_pv / 1.7
4. Calculate beam waist: w₀ = √(z₀λ/π)

### Data Normalization

Standard practice divides closed aperture by open aperture signals (T_CA / T_OA) to isolate refractive effects. This implementation uses **baseline detrending** instead:

- The open aperture data is ignored due to detector artifacts
- A linear baseline is fit to the far-from-focus regions
- The closed aperture signal is normalized to this baseline (set to 1.0)
- This removes laser drift and sample tilt effects

### Nonlinear Index Extraction

The code determines n₂ through **iterative slope matching**:

1. **Experimental slope**: Calculate the linear relationship between peak-valley transmission difference (ΔT_pv) and incident peak power
2. **Numerical simulation**: For trial n₂ values, simulate beam propagation through the sample using the split-step Fourier method
3. **Optimization**: Adjust n₂ until the simulated slope matches experimental data within tolerance (<1%)

The simulation accounts for the full nonlinear phase accumulation rather than relying on thin-sample approximations.

---

## Key Modifications from Original Code

| Aspect | Original Implementation | This Version |
|--------|------------------------|--------------|
| **Target materials** | Semiconductors with TPA | Transparent dielectrics |
| **Absorption handling** | Fits β from open aperture | Sets β = 0, bypasses fitting |
| **Beam waist** | Extracted from OA fit | Geometric calculation from CA peak-valley |
| **Normalization** | T_CA / T_OA | T_CA / linear baseline trend |
| **Peak power** | Calculated from rep rate | Manual override from lab measurements |
| **Visualization** | Standard Matplotlib | Interactive Bokeh plots with metric units |

---

## Usage Instructions

### Data Preparation

Create `Z_Scan_Data.csv` in the working directory with columns:
```
Z_pos, Power_OA, Power_CA, Z_pos, Power_OA, Power_CA, ...
```
Each triplet represents one power level measurement.

### Configuration

Edit the user settings block in the script:

```python
L_setup = 300  # mm, lens to aperture distance
f_lens = 150   # mm, focal length
manual_peak_powers = [P1, P2, P3, ...]  # Watts, from lab log
```

### Execution

Run the script and respond to interactive prompts:

1. **Linear regions**: Specify Z-ranges far from focus for baseline fitting (e.g., "0-5" and "15-20" mm)
2. **Nonlinearity sign**: Enter "0" for self-focusing (positive n₂)
3. **Analysis window**: Define Z-range containing peak and valley (e.g., "5-15" mm)

### Output

The program generates:
- Extracted n₂ value with confidence metrics
- Interactive plots showing experimental data, fits, and simulations
- Diagnostic information about beam parameters and phase shifts

---

## Dependencies

```
numpy
pandas
scipy
bokeh
lmfit
scikit-learn
```

Install via: `pip install numpy pandas scipy bokeh lmfit scikit-learn`

---

## References

1. M. Sheik-Bahae et al., "Sensitive Measurement of Optical Nonlinearities Using a Single Beam," *IEEE J. Quantum Electron.* **26**, 760-769 (1990)
2. P. Kabaciński et al., "Nonlinear refractive index measurement by SPM-induced phase regression," *Opt. Express* **27**, 11018-11032 (2019)
3. Original framework: [J.J. Arango - Z-scan Parameter Extraction](https://github.com/juanjosearango/z-scan-parameter-extraction-program)

---

## Technical Notes

- The code assumes Gaussian beam propagation and cubic optical nonlinearity
- Far-field approximation requires L_setup >> z₀
- For best results, ensure the scan range covers at least ±2z₀ from focus
- Peak power values should reflect actual pulse characteristics, not nominal specifications
