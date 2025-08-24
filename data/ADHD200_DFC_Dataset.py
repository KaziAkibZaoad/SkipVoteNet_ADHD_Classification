import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def _safe_load(path, key_npy=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        if key_npy is None:
            for k in ("dFC", "sFC", "arr_0"):
                if k in data:
                    return data[k]
            raise KeyError(f"No suitable key found in {path}. Available: {list(data.keys())}")
        return data[key_npy]
    else:
        raise ValueError(f"Unsupported file type: {path}")

class ADHD200DFCDataset(Dataset):
    """
    PyTorch dataset yielding:
        x: float32 tensor-like [K, 1, H, W]  (dFC)
        y: int label in {0,1,2} -> {TDC, ADHDC, ADHDI}
        sid: subject id (str)
        optional sFC: [N, N] if available
    """

    def __init__(
        self,
        phenotype_csv: str,
        fc_root_dir: str,
        k_segments: int = 7,
        include_hyperactive: bool = False,
        subject_ids_subset=None,
    ):
        super().__init__()
        assert os.path.isfile(phenotype_csv), f"Missing {phenotype_csv}"
        assert os.path.isdir(fc_root_dir), f"Missing {fc_root_dir}"

        # Subfolders
        dfc_dir = os.path.join(fc_root_dir, "dFC_K7")
        sfc_dir = os.path.join(fc_root_dir, "sFC")

        self.df = pd.read_csv(phenotype_csv)

        # Check SubjectID column
        if "SubjectID" not in self.df.columns:
            raise ValueError("phenotype.csv must contain a 'SubjectID' column")
        self.sid_col = "SubjectID"

        # Check RunID column
        if "RunID" not in self.df.columns:
            raise ValueError("phenotype.csv must contain a 'RunID' column")
        self.run_col = "RunID"

        # DX: 0=TDC, 1=ADHDC, 2=ADHDH (exclude), 3=ADHDI
        if not include_hyperactive:
            self.df = self.df[self.df["DX"] != 2]
        self.df.loc[self.df["DX"] == 3, "DX"] = 2  # recode 3->2

        # Optional subset of subjects
        if subject_ids_subset is not None:
            subset = set(map(str, subject_ids_subset))
            self.df = self.df[self.df[self.sid_col].astype(str).isin(subset)]

        # List all available dFC and sFC files
        all_dfc_files = glob.glob(os.path.join(dfc_dir, "*.npy")) + glob.glob(os.path.join(dfc_dir, "*.npz"))
        present_dfc = set(os.path.splitext(os.path.basename(p))[0] for p in all_dfc_files)

        all_sfc_files = glob.glob(os.path.join(sfc_dir, "*.npy")) + glob.glob(os.path.join(sfc_dir, "*.npz"))
        present_sfc = set(os.path.splitext(os.path.basename(p))[0] for p in all_sfc_files)

        keep_rows = []
        dfc_paths = []
        sfc_paths = []

        for _, row in self.df.iterrows():
            sid_raw = str(row[self.sid_col]).zfill(7)
            run_id = str(row[self.run_col])
            file_base = f"{sid_raw}_{run_id}"

            # dFC file
            npy_dfc = os.path.join(dfc_dir, file_base + "_dFC.npy")
            npz_dfc = os.path.join(dfc_dir, file_base + "_dFC.npz")
            dfc_path = npy_dfc if os.path.isfile(npy_dfc) else npz_dfc if os.path.isfile(npz_dfc) else None
            if dfc_path is None:
                continue

            # optional sFC file
            npy_sfc = os.path.join(sfc_dir, file_base + "_sFC.npy")
            npz_sfc = os.path.join(sfc_dir, file_base + "_sFC.npz")
            sfc_path = npy_sfc if os.path.isfile(npy_sfc) else npz_sfc if os.path.isfile(npz_sfc) else None

            # load dFC to check shape
            arr = _safe_load(dfc_path, key_npy="dFC")
            if arr.ndim != 3:
                raise ValueError(f"dFC for {file_base} must be [K,N,N], got {arr.shape}")
            if arr.shape[0] != k_segments:
                pass  # optional: flexible K

            keep_rows.append(row)
            dfc_paths.append(dfc_path)
            sfc_paths.append(sfc_path)

        self.meta = pd.DataFrame(keep_rows).reset_index(drop=True)
        self.dfc_paths = dfc_paths
        self.sfc_paths = sfc_paths

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        sid = str(row[self.sid_col])
        y = int(row["DX"])
        dfc = _safe_load(self.dfc_paths[idx], key_npy="dFC")
        x = dfc[:, None, :, :].astype(np.float32).squeeze(1)  # add channel dim [K,1,N,N]

        sfc = None
        if self.sfc_paths[idx] is not None:
            sfc = _safe_load(self.sfc_paths[idx], key_npy="sFC").astype(np.float32)

        return x, y, sid, sfc

    def subjects(self):
        return list(self.meta[self.sid_col].astype(str).values)
