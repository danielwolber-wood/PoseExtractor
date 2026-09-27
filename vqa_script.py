
import math
import torch
import pyiqa
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# --- Configuration & Constants ---
WORK_DIR = "./Data/Extracted"
OUTPUT_CSV = "quality_output.csv"

# Define once globally to save memory during loop execution
STANDARD_RATIOS = {
    '16:9': 1.777,
    '3:2': 1.5,
    '4:3': 1.333,
    '5:4': 1.25,
    '1:1': 1.0,
    '4:5': 0.8,
    '3:4': 0.75,
    '2:3': 0.667,
    '9:16': 0.562
}

def get_closest_ratio(val):
    """Matches a decimal aspect ratio to the closest standard photographic ratio."""
    closest_name = min(STANDARD_RATIOS.keys(), key=lambda k: abs(STANDARD_RATIOS[k] - val))

    # Set a threshold. If it's too far from a standard, call it 'Custom'
    if abs(STANDARD_RATIOS[closest_name] - val) > 0.05:
        return 'Custom'
    return closest_name

def main():
    torch.hub.set_dir("./torch_dir")
    workdir = Path(WORK_DIR)

    # Ensure directory exists before searching
    if not workdir.exists():
        print(f"Error: Directory {workdir} not found.")
        return

    files = list(workdir.rglob("*.jpg"))
    if not files:
        print(f"No JPG files found in {workdir}.")
        return

    # Set up device with a fallback for non-Apple Silicon machines
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize metrics
    metric_names = [
        'align-one', 'musiq', 'hyperiqa', 'nima', 'brisque',
        'clipiqa', 'niqe', 'arniqa', 'liqe'
    ]

    iqa_metrics = {}
    for name in metric_names:
        print(f"Loading metric: {name}")
        try:
            iqa_metrics[name] = pyiqa.create_metric(name, device=device)
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    transform = transforms.ToTensor() # Initialize transform once outside the loop

    scores = []

    # desc parameter adds a nice label to the progress bar
    for file in tqdm(files, desc="Scoring Images"):

        try:
            # 1. Load and process image
            img_pil = Image.open(file).convert('RGB')
            width, height = img_pil.size

            # 2. Compute Visual Scores (Wrapped in no_grad for speed/memory efficiency)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            metric_results = {}
            with torch.no_grad():
                for name, metric in iqa_metrics.items():
                    try:
                        raw_score = metric(img_tensor).item()
                        metric_results[f"{name}_raw"] = raw_score

                        if hasattr(metric, 'score_range') and metric.score_range is not None:
                            # example "~0, ~100"
                            score_range = metric.score_range
                            l = score_range.split(", ")
                            items = [int(f.replace("~","")) for f in l]
                            min_val = items[0]
                            max_val = items[1]
                            #min_val, max_val = metric.score_range
                            if max_val > min_val:
                                scaled = (raw_score - min_val) / (max_val - min_val)
                                if getattr(metric, 'lower_better', False):
                                    scaled = 1.0 - scaled
                                metric_results[f"{name}_scaled"] = scaled
                    except Exception as e:
                        tqdm.write(f"Error computing {name} for {file.name}: {e}")

            # 3. Resolution Processing
            gcd = math.gcd(width, height)
            ratio_w = width // gcd
            ratio_h = height // gcd

            # 4. Append flat data dictionary
            data_dict = {
                "file_path": str(file),
                "file_name": file.name,
                "width": width,
                "height": height,
                "aspect_ratio": f"{ratio_w}:{ratio_h}",
                "decimal": round(width / height, 2)
            }
            data_dict.update(metric_results)
            scores.append(data_dict)

        except Exception as e:
            # Catch corrupt images so they don't crash the entire hours-long loop
            tqdm.write(f"Error processing {file.name}: {e}")

    # --- Dataframe Processing ---
    print("\nProcessing data and generating CSV...")
    df = pd.DataFrame(scores)

    # Calculate pixel stats
    df['total_pixels'] = df['width'] * df['height']
    df['megapixels'] = df['total_pixels'] / 1_000_000

    # Determine orientation
    conditions = [
        (df['width'] > df['height']),
        (df['width'] < df['height']),
        (df['width'] == df['height'])
    ]
    choices = ['Landscape', 'Portrait', 'Square']
    df['orientation'] = np.select(conditions, choices, default='Unknown')

    # Categorize ratios
    df['standard_ratio_bucket'] = df['decimal'].apply(get_closest_ratio)

    # Save to disk
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved results to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
