"""Label cleaner and binary mapper for Elliptic Bitcoin transactions."""

from typing import Dict, Union
import numpy as np
import pandas as pd

from src.common.logger import get_logger

logger = get_logger("Module2.LabelCleaner")

# Canonical label mapping
# '1' -> 1 (illicit / positive fraud class)
# '2' -> 0 (licit / negative legitimate class)
# 'unknown' -> -1 (unlabeled / background graph node)
LABEL_MAPPING: Dict[str, int] = {
    "1": 1,
    "2": 0,
    "unknown": -1,
}

REVERSE_LABEL_MAPPING: Dict[int, str] = {
    1: "illicit",
    0: "licit",
    -1: "unknown",
}


class LabelCleaner:
    """Standardizes transaction class labels into binary supervised targets."""

    @staticmethod
    def map_label(raw_label: Union[str, int]) -> int:
        """
        Maps a single raw class string/int to standardized integer target:
        '1' / 1 -> 1 (illicit)
        '2' / 2 -> 0 (licit)
        'unknown' / -1 -> -1 (unlabeled)
        """
        key = str(raw_label).strip()
        if key in LABEL_MAPPING:
            return LABEL_MAPPING[key]
        if key == "-1":
            return -1
        if key == "0":
            return 0
        raise ValueError(f"Invalid label encounter: '{raw_label}'. Expected '1', '2', or 'unknown'.")

    def clean_labels(self, df_classes: pd.DataFrame) -> pd.DataFrame:
        """
        Takes raw classes DataFrame (columns ['txId', 'class']) and returns
        DataFrame with ['txId', 'raw_class', 'label', 'is_labeled'].
        """
        if "txId" not in df_classes.columns or "class" not in df_classes.columns:
            raise ValueError("Input DataFrame must contain 'txId' and 'class' columns.")

        logger.info(f"Cleaning labels for {len(df_classes)} transactions...")
        
        df = df_classes.copy()
        df["raw_class"] = df["class"].astype(str)
        
        # Map labels
        df["label"] = df["raw_class"].map(lambda x: self.map_label(x)).astype(np.int32)
        df["is_labeled"] = df["label"] != -1

        illicit_cnt = (df["label"] == 1).sum()
        licit_cnt = (df["label"] == 0).sum()
        unknown_cnt = (df["label"] == -1).sum()

        logger.info(
            f"Label cleaning complete: Illicit(1)={illicit_cnt}, Licit(0)={licit_cnt}, Unknown(-1)={unknown_cnt}, Labeled={illicit_cnt + licit_cnt}"
        )
        return df[["txId", "raw_class", "label", "is_labeled"]]
