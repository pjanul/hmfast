import os
import requests

COSMOPOWER_MODELS = {
    "ede-v2": [
        "growth-and-distances/DAZ_v2.npz",
        "growth-and-distances/HZ_v2.npz",
        "growth-and-distances/S8Z_v2.npz",
        "derived-parameters/DER_v2.npz",
        "PK/PKL_v2.npz",
        "PK/PKNL_v2.npz",
        "TTTEEE/TT_v2.npz",
        "TTTEEE/TE_v2.npz",
        "TTTEEE/EE_v2.npz",
        "PP/PP_v2.npz",
        "BB/BB_v2.npz",
    ],
    "ede-v1": [
        "growth-and-distances/DAZ_v1.npz",
        "growth-and-distances/HZ_v1.npz",
        "growth-and-distances/S8Z_v1.npz",
        "derived-parameters/DER_v1.npz",
        "PK/PKL_v1.npz",
        "PK/PKNL_v1.npz",
        "TTTEEE/TT_v1.npz",
        "TTTEEE/TE_v1.npz",
        "TTTEEE/EE_v1.npz",
        "PP/PP_v1.npz",
    ],
    "lcdm": [
        "growth-and-distances/DAZ_v1.npz",
        "growth-and-distances/HZ_v1.npz",
        "growth-and-distances/S8Z_v1.npz",
        "derived-parameters/DER_v1.npz",
        "PK/PKL_v1.npz",
        "PK/PKNL_v1.npz",
        "TTTEEE/TT_v1.npz",
        "TTTEEE/TE_v1.npz",
        "TTTEEE/EE_v1.npz",
        "PP/PP_v1.npz",
    ],
    "mnu": [
        "growth-and-distances/DAZ_mnu_v1.npz",
        "growth-and-distances/HZ_mnu_v1.npz",
        "growth-and-distances/S8Z_mnu_v1.npz",
        "derived-parameters/DER_mnu_v1.npz",
        "PK/PKL_mnu_v1.npz",
        "PK/PKNL_mnu_v1.npz",
        "TTTEEE/TT_mnu_v1.npz",
        "TTTEEE/TE_mnu_v1.npz",
        "TTTEEE/EE_mnu_v1.npz",
        "PP/PP_mnu_v1.npz",
    ],
    "neff": [
        "growth-and-distances/DAZ_neff_v1.npz",
        "growth-and-distances/HZ_neff_v1.npz",
        "growth-and-distances/S8Z_neff_v1.npz",
        "derived-parameters/DER_neff_v1.npz",
        "PK/PKL_neff_v1.npz",
        "PK/PKNL_neff_v1.npz",
        "TTTEEE/TT_neff_v1.npz",
        "TTTEEE/TE_neff_v1.npz",
        "TTTEEE/EE_neff_v1.npz",
        "PP/PP_neff_v1.npz",
    ],
    "wcdm": [
        "growth-and-distances/DAZ_w_v1.npz",
        "growth-and-distances/HZ_w_v1.npz",
        "growth-and-distances/S8Z_w_v1.npz",
        "derived-parameters/DER_w_v1.npz",
        "PK/PKL_w_v1.npz",
        "PK/PKNL_w_v1.npz",
        "TTTEEE/TT_w_v1.npz",
        "TTTEEE/TE_w_v1.npz",
        "TTTEEE/EE_w_v1.npz",
        "PP/PP_w_v1.npz",
    ],
    "mnu-3states": [
        "growth-and-distances/DAZ_v1.npz",
        "growth-and-distances/HZ_v1.npz",
        "growth-and-distances/S8Z_v1.npz",
        "derived-parameters/DER_v1.npz",
        "PK/PKL_v1.npz",
        "PK/PKNL_v1.npz",
        "TTTEEE/TT_v1.npz",
        "TTTEEE/TE_v1.npz",
        "TTTEEE/EE_v1.npz",
        "PP/PP_v1.npz",
    ],
}

AUX_FILES = [
    {
        "url": "https://raw.githubusercontent.com/CLASS-SZ/class_sz/master/class_sz/class_sz_auxiliary_files/includes/normalised_dndz_cosmos_0.txt",
        "subdir": "auxiliary_files",
        "filename": "normalised_dndz_cosmos_0.txt"
    },
    {
        "url": "https://raw.githubusercontent.com/CLASS-SZ/class_sz/master/class-sz/class_sz_auxiliary_files/includes/nz_lens_bin1.txt",
        "subdir": "auxiliary_files",
        "filename": "nz_lens_bin1.txt"
    },
    {
        "url": "https://raw.githubusercontent.com/CLASS-SZ/class_sz/master/class-sz/class_sz_auxiliary_files/includes/nz_source_normalized_bin4.txt",
        "subdir": "auxiliary_files",
        "filename": "nz_source_normalized_bin4.txt"
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
        #print(f"  Already exists: {local_path} (skipped)")
        return
    #print(f"  Downloading: {url} → {local_path}")
    
    headers = {"User-Agent": "python-requests/2.0"}  # GitHub sometimes needs this
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(response.content)
    #print(f"  Saved to {local_path}")

def download_emulators(models="all", skip_existing=True):
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


    for model in models_to_fetch:
        subdir = "ede" if model.startswith("ede") else model

        # NEW: Only print if files might be downloaded
        if not skip_existing:
            print(f"Downloading cosmopower model: {model}")
        else:
            # Check if any file is missing
            needs_download = False
            for rel_path in COSMOPOWER_MODELS[model]:
                local_dir = os.path.join(target_dir, subdir, os.path.dirname(rel_path))
                local_path = os.path.join(local_dir, os.path.basename(rel_path))
                if not os.path.exists(local_path):
                    needs_download = True
                    break
            if needs_download:
                print(f"hmfast: downloading cosmopower model {model}")

        for rel_path in COSMOPOWER_MODELS[model]:
            url = f"https://github.com/cosmopower-organization/{subdir}/raw/main/{rel_path}"
            local_dir = os.path.join(target_dir, subdir, os.path.dirname(rel_path))
            local_path = os.path.join(local_dir, os.path.basename(rel_path))
            download_file(url, local_path, skip_existing=skip_existing)

    # Auxiliary files section
    aux_dir_root = os.path.join(target_dir, "auxiliary_files")

    # NEW: Only print if at least one aux file is missing or skip_existing=False
    if not skip_existing:
        print(f"Downloading auxiliary files")
    else:
        needs_aux = any(
            not os.path.exists(os.path.join(target_dir, aux["subdir"], aux["filename"]))
            for aux in AUX_FILES
        )
        if needs_aux:
            print(f"hmfast: downloading auxiliary files")

    for aux in AUX_FILES:
        aux_dir = os.path.join(target_dir, aux["subdir"])
        aux_path = os.path.join(aux_dir, aux["filename"])
        download_file(aux["url"], aux_path, skip_existing=skip_existing)

