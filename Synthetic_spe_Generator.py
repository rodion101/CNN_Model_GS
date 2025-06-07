"""
This code is written to generate .spe files (ASCII data with specific noise characteristics) from a template
this is a part of a project to build ML model for spectrum analysis by A.AbuAli
"""
##############################################################################################################
import os
import numpy as np
from pathlib import Path

#The following function read a .spe file in the directory and return the raw counts array
def read_counts(spe_path):
    with open(spe_path, 'r', errors='ignore') as f:
        lines = f.readlines()

    data_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith('$DATA'):
            data_idx = i
            break
    if data_idx is None:
        raise ValueError(f"No '$DATA' found in {spe_path}")

    start_ch, end_ch = map(int, lines[data_idx + 1].split())
    num_channels = end_ch - start_ch + 1

    counts_lines = lines[data_idx + 2 : data_idx + 2 + num_channels]
    if len(counts_lines) < num_channels:
        raise ValueError(f"Expected {num_channels} counts in {spe_path}")

    counts = np.array([int(l.strip()) for l in counts_lines], dtype=int)
    return counts
# the following function Return the channel index of the highest local maximum within +/- neighborhood.
def detect_main_peak(counts, neighborhood=5):
    # Simple: take absolute max, then confirm in neighborhood
    idx = counts.argmax()
    left = max(idx - neighborhood, 0)
    right = min(idx + neighborhood + 1, len(counts))
    local_max_idx = left + counts[left:right].argmax()
    return local_max_idx

"""
    the following function Builds one synthetic spectrum from a template:
      - noise_scale: multiply template by this before Poisson
      - peak_jitter_max: ±channels to jitter the main peak
      - width_range: tuple(min_sigma, max_sigma) for Gaussian
      - intensity_var: fraction to vary peak intensity ±20%
      - background_max: uniform background up to this many counts per channel
"""
def generate_one_spectrum(template_counts,
                          main_peak,
                          noise_scale=1.0,
                          peak_jitter_max=5,
                          width_range=(2, 6),
                          intensity_var=0.2,
                          background_max=5):
##############################################################################
    num_ch = len(template_counts)
    # 1) Scale & Poisson‐noise the entire template
    scaled = np.clip(template_counts.astype(float) * noise_scale, 0, None)
    base = np.random.poisson(scaled).astype(float)

    # 2) Determine this spectrum’s main peak position and parameters
    jitter = np.random.randint(-peak_jitter_max, peak_jitter_max + 1)
    peak_pos = np.clip(main_peak + jitter, 0, num_ch - 1)

    # Random width (sigma) in channels
    sigma = np.random.uniform(width_range[0], width_range[1])
    # Random peak intensity multiplier
    orig_peak_height = template_counts[main_peak]
    peak_intensity = orig_peak_height * np.random.uniform(1 - intensity_var,
                                                          1 + intensity_var)

    # Build a Gaussian curve
    x = np.arange(num_ch)
    gauss = np.exp(-0.5 * ((x - peak_pos) / sigma) ** 2)
    gauss *= peak_intensity / gauss.sum()

    # 3) Add Gaussian peak to the baseline
    combined = base + gauss

    # 4) Add a small uniform background floor
    background = np.random.uniform(0, background_max, size=num_ch)
    combined += background

    # 5) Final Poisson sampling
    final_counts = np.random.poisson(np.clip(combined, 0, None)).astype(int)
    return final_counts

def write_valid_spe(out_path, counts):
    with open(out_path, 'w') as f:
        f.write("$DATA:\n")
        f.write(f"0 {len(counts) - 1}\n")
        for c in counts:
            f.write(f"{c}\n")

def generate_batch(template_dir,
                     output_dir,
                     n_per_class=500,
                     gain_shift_range=(0.95, 1.05),
                     peak_jitter_max=5,
                     width_range=(2, 6),
                     intensity_var=0.2,
                     background_max=5):
    """
    how this works? For each class folder in template_dir (each containing exactly one template .spe),
    we generate n_per_class synthetic spectra with aggressive augmentation.

    Arguments:
      - template_dir: path containing subfolders e.g. 'Depleted_Uranium', each with 1 .spe
      - output_dir: where to write folders of synthetic spectra per class
      - n_per_class: how many to generate per class
      - gain_shift_range: (min, max) uniform range for overall gain shift
      - peak_jitter_max: ±chan jitter for main peak
      - width_range: (min_sigma, max_sigma) for Gaussian width
      - intensity_var: fraction ± to vary peak height
      - background_max: max uniform background per channel
    """
    os.makedirs(output_dir, exist_ok=True)
    for class_folder in sorted(Path(template_dir).iterdir()):
        if not class_folder.is_dir():
            continue
        # Find the one .spe inside
        spe_files = list(class_folder.glob("*.spe"))
        if len(spe_files) != 1:
            print(f"Skipping {class_folder}: expected exactly 1 .spe, found {len(spe_files)}")
            continue

        cls_name = class_folder.name
        print(f"Processing class '{cls_name}'")
        class_out = Path(output_dir) / cls_name
        class_out.mkdir(parents=True, exist_ok=True)

        # Read the template counts and detect main peak
        tpl_counts = read_counts(str(spe_files[0]))
        main_peak = detect_main_peak(tpl_counts)

        for i in range(1, n_per_class + 1):
            gain = np.random.uniform(gain_shift_range[0], gain_shift_range[1])
            synth = generate_one_spectrum(
                template_counts=tpl_counts,
                main_peak=main_peak,
                noise_scale=gain,
                peak_jitter_max=peak_jitter_max,
                width_range=width_range,
                intensity_var=intensity_var,
                background_max=background_max
            )
            out_name = f"{cls_name}_{i:04d}.spe"
            write_valid_spe(str(class_out / out_name), synth)


        print(f"  → Generated {n_per_class} synthetic files for '{cls_name}'")

if __name__ == "__main__":
    template_dir = r"C:\Users\Rodion\Downloads\Pu-syn"
    output_dir   = r"C:\Users\Rodion\Downloads\Pu-spe"

    generate_batch(
        template_dir      = template_dir,  #this is the parent directory containing the sub folders
        output_dir        = output_dir,
        n_per_class       = 1000,          #this is the number of synthetic files to be generated
        gain_shift_range  = (0.95, 1.05),  # ±5% gain variation
        peak_jitter_max   = 5,             # ±5 channels
        width_range       = (2, 6),        # sigma between 2–6 channels
        intensity_var     = 0.25,          # ±25% peak intensity
        background_max    = 10             # adds up to 10 counts of uniform background
    )
#######################################################################################################