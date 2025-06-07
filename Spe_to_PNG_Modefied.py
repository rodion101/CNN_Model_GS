import os
import re
import matplotlib.pyplot as plt

def read_spe_file(filepath):
    """
    Reads a Genie-2000 ASCII .spe file.
    Assumes the 4th line (index 3) is the 'material type' label.
    Returns a tuple (spec_id, spectrum_counts_list).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Get material type from line 4
    raw_label = lines[3].strip()
    # If it’s like "$MATERIAL_TYPE: UO2", drop the tag
    if ':' in raw_label:
        spec_id = raw_label.split(':', 1)[1].strip()
    else:
        spec_id = raw_label

    # Now grab the spectrum data
    spectrum = []
    data_started = False
    for line in lines:
        if data_started:
            if line.strip().startswith('$'):
                break
            spectrum.extend(int(x) for x in line.split())
        elif line.upper().strip() == "$DATA:":
            data_started = True

    return spec_id, spectrum

def sanitize_filename(name):
    """
    Replace spaces with underscores and strip invalid filename chars.
    """
    name = name.replace(' ', '_')
    return re.sub(r'[^A-Za-z0-9_\-]', '', name)

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith('.spe'):
            continue

        path_in = os.path.join(input_folder, fname)
        spec_id, spectrum = read_spe_file(path_in)

        # Build safe filename
        safe_id = sanitize_filename(spec_id)
        out_png = os.path.join(output_folder, f"{safe_id}.png")

        # Plot & save
        plt.figure(figsize=(10, 5))
        plt.plot(spectrum)
        plt.title(spec_id)
        plt.xlabel("Channel")
        plt.ylabel("Counts")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

        print(f"[✓] {fname} → {safe_id}.png")

if __name__ == "__main__":
    input_folder = r"C:\Users\Rodion\Downloads\U-235 focused Spe"
    output_folder = r"C:\Users\Rodion\Downloads\U-235 focused PNG"
    process_folder(input_folder, output_folder)
