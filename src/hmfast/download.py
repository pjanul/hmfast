import os
import requests

COSMOPOWER_MODELS = {
    "ede-v2": [
        "growth-and-distances/DAZ_v2.npz",
        "growth-and-distances/HZ_v2.npz",
        "growth-and-distances/S8Z_v2.npz",
        "PK/PKL_v2.npz",
        "PK/PKNL_v2.npz",
    ],
    "ede-v1": [
        "growth-and-distances/DAZ_v1.npz",
        "growth-and-distances/HZ_v1.npz",
        "growth-and-distances/S8Z_v1.npz",
        "PK/PKL_v1.npz",
        "PK/PKNL_v1.npz",
    ],
    "lcdm": [
        "growth-and-distances/DAZ_v1.npz",
        "growth-and-distances/HZ_v1.npz",
        "growth-and-distances/S8Z_v1.npz",
        "PK/PKL_v1.npz",
        "PK/PKNL_v1.npz",
    ],
    "mnu": [
        "growth-and-distances/DAZ_mnu_v1.npz",
        "growth-and-distances/HZ_mnu_v1.npz",
        "growth-and-distances/S8Z_mnu_v1.npz",
        "PK/PKL_mnu_v1.npz",
        "PK/PKNL_mnu_v1.npz",
    ],
    "neff": [
        "growth-and-distances/DAZ_neff_v1.npz",
        "growth-and-distances/HZ_neff_v1.npz",
        "growth-and-distances/S8Z_neff_v1.npz",
        "PK/PKL_neff_v1.npz",
        "PK/PKNL_neff_v1.npz",
    ],
    "wcdm": [
        "growth-and-distances/DAZ_w_v1.npz",
        "growth-and-distances/HZ_w_v1.npz",
        "growth-and-distances/S8Z_w_v1.npz",
        "PK/PKL_w_v1.npz",
        "PK/PKNL_w_v1.npz",
    ],
    "mnu-3states": [
        "growth-and-distances/DAZ_v1.npz",
        "growth-and-distances/HZ_v1.npz",
        "growth-and-distances/S8Z_v1.npz",
        "PK/PKL_v1.npz",
        "PK/PKNL_v1.npz",
    ],
}

AUX_FILES = [
    {
        "url": "https://raw.githubusercontent.com/CLASS-SZ/class_sz/master/class-sz/class_sz_auxiliary_files/includes/normalised_dndz_cosmos_0.txt",
        "subdir": "auxiliary_files",
        "filename": "normalised_dndz_cosmos_0.txt"
    }
]

def get_default_data_path():
    """
    Returns the base data path for emulator and auxiliary files.
    Uses the HMFAST_DATA_PATH environment variable if set,
    otherwise defaults to ~/hmfast_data.
    """
    return os.environ.get("HMFAST_DATA_PATH", os.path.join(os.path.expanduser("~"), "hmfast_data"))

def download_file(url, local_path, skip_existing=True):
    if skip_existing and os.path.exists(local_path):
        print(f"  Already exists: {local_path} (skipped)")
        return
    print(f"  Downloading: {url} → {local_path}")
    
    headers = {"User-Agent": "python-requests/2.0"}  # GitHub sometimes needs this
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(response.content)
    print(f"  Saved to {local_path}")

def download_emulators(models="ede-v2", skip_existing=True):
    """
    Download emulator .npz files and all required auxiliary files
    into the default data directory.

    Files are always downloaded to ~/hmfast_data, or to the path specified by the
    environment variable HMFAST_DATA_PATH.
    """
    target_dir = get_default_data_path()
    os.makedirs(target_dir, exist_ok=True)

    valid_models = list(COSMOPOWER_MODELS.keys())

    if models == "all" or models == ["all"]:
        models_to_fetch = valid_models
    elif models is None:
        models_to_fetch = ["ede-v2"]
    elif isinstance(models, str):
        models_to_fetch = [models]
    else:
        models_to_fetch = models

    for m in models_to_fetch:
        if m not in valid_models:
            raise ValueError(f"Unknown model '{m}'. Available: {valid_models}")

    print(f"Downloading emulators to: {target_dir}")
    for model in models_to_fetch:
        subdir = "ede" if model.startswith("ede") else model
        print(f"Downloading cosmopower model: {model} → {subdir}/")
        for rel_path in COSMOPOWER_MODELS[model]:
            url = f"https://github.com/cosmopower-organization/{subdir}/raw/main/{rel_path}"
            local_dir = os.path.join(target_dir, subdir, os.path.dirname(rel_path))
            local_path = os.path.join(local_dir, os.path.basename(rel_path))
            download_file(url, local_path, skip_existing=skip_existing)

    # Always download auxiliary files
    print(f"Downloading auxiliary files to: {target_dir}/auxiliary_files")
    for aux in AUX_FILES:
        aux_dir = os.path.join(target_dir, aux["subdir"])
        aux_path = os.path.join(aux_dir, aux["filename"])
        download_file(aux["url"], aux_path, skip_existing=skip_existing)

    print("Download complete.")