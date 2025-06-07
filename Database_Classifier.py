import os


def classify_u_enrichment(file_path):
    """
    Reads the .spe file at file_path, locates the '235U' line,
    and returns the enrichment value as a float.
    """
    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('235U'):
                parts = stripped.split()
                try:
                    enrichment_value = float(parts[1])
                    return enrichment_value
                except (IndexError, ValueError):
                    raise ValueError(f"Unable to parse enrichment value from: '{line.strip()}'")
    raise ValueError(f"'235U' line not found in file: {file_path}")


def classify_files_by_enrichment(input_directory, output_directories):
    """
    Scans all .spe files in input_directory, classifies them based on 235U enrichment,
    and moves each file into the corresponding directory in output_directories.

    output_directories should be a dict with keys: "Depleted", "Natural", "Enriched"
    and values as the paths (strings) to where you want those files moved.
    """
    # 1) Ensure all output subfolders exist
    for category, out_dir in output_directories.items():
        os.makedirs(out_dir, exist_ok=True)

    # 2) Loop over each .spe file
    for filename in os.listdir(input_directory):
        if not filename.lower().endswith('.spe'):
            continue  # skip non-.spe files

        file_path = os.path.join(input_directory, filename)
        try:
            enrichment = classify_u_enrichment(file_path)
        except ValueError as e:
            print(f"Skipping '{filename}': {e}")
            continue

        # 3) Decide category by simple thresholds
        if enrichment < 0.7:
            category = "Depleted"
        elif 0.69 <= enrichment <= 0.75:
            category = "Natural"
        else:
            category = "Enriched"

        # 4) Move the file into the matching subfolder
        destination = os.path.join(output_directories[category], filename)
        os.replace(file_path, destination)
        print(f"Moved '{filename}' → '{category}'  (235U = {enrichment:.4f}%)")


# ─── Example usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Change this to the folder where your .spe files currently live
    input_dir = r"C:\Users\Rodion\Desktop\Uranium_Database"

    # Adjust these three to wherever you’d like the classified files to go
    output_dirs = {
        "Depleted": r"C:\Users\Rodion\Desktop\Spectra_Raw_Database\Depleted_Uranium",
        "Natural": r"C:\Users\Rodion\Desktop\Spectra_Raw_Database\Natural_Uranium",
        "Enriched": r"C:\Users\Rodion\Desktop\Spectra_Raw_Database\Enriched_Uranium"
    }

    classify_files_by_enrichment(input_dir, output_dirs)
