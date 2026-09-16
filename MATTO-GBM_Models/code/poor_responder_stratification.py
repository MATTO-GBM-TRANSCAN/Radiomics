import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from matto_radiomics.utils import *

# Ground truth definition from the paper (section 3.3): a patient is a "poor
# responder" if they have short Overall Survival (below the training-set median,
# less than 262 days) and/or Acute Recurrence (progression within 90 days). This
# script combines the OS threshold classification (see overall_survival_threshold.py)
# with an Acute Recurrence model's predictions to reproduce that stratification.
#
# The paper's headline result (Sensitivity=82%, Specificity=75%, Table 6:
# "OS+AR(FET-PET+T1wCE)") combines this OS rule with the AcuteRecurrence-MR+PET-GTV
# model specifically, not the AcuteRecurrence-MR-MRandPET+Clinical one. A second
# combination (with the MR-and-PET intersection AR signature) was also tried in the
# paper but had lower specificity (50% of good responders misclassified vs. 25%).


def stratify_responders(
    os_predictions: pd.DataFrame,
    ar_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine an OS classification and an Acute Recurrence (AR) classification into a
    single "Good responder" / "Poor responder" label, per the paper's definition.

    :param os_predictions: Output of `overall_survival_threshold.py`
        (`classify_overall_survival`), indexed by patient id, with a
        "Predicted OS Group" column ("Long OS" / "Short OS").
    :type os_predictions: pandas.DataFrame
    :param ar_predictions: Output of `inference.py`'s `event_inference` for an Acute
        Recurrence model, indexed by patient id, with a "Predicted Label" column
        (1 = Acute Recurrence, 0 = no Acute Recurrence).
    :type ar_predictions: pandas.DataFrame
    :return: DataFrame indexed by the patients present in both inputs, with the OS
        group, the AR label, and the combined "Poor responder" / "Good responder"
        classification.
    :rtype: pandas.DataFrame
    """
    combined = os_predictions[["Predicted OS Group"]].join(
        ar_predictions[["Predicted Label"]], how="inner"
    )
    combined = combined.rename(columns={"Predicted Label": "Predicted AR"})

    is_poor_responder = (combined["Predicted OS Group"] == "Short OS") | (
        combined["Predicted AR"] == 1
    )
    combined["Stratification"] = is_poor_responder.map(
        {True: "Poor responder", False: "Good responder"}
    )

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Combine an Overall Survival threshold classification and an Acute "
            "Recurrence model's predictions into a Good/Poor responder label, per "
            "the paper's definition."
        )
    )
    parser.add_argument("config_file", help="Path to the JSON configuration file.")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    id_column = config["id_column"]
    experiment_name = config.get("experiment_name", "Poor_responder_stratification")
    root_dir = Path(config["root_dir"])
    os_predictions_path = root_dir / config["os_predictions"]
    ar_predictions_path = root_dir / config["ar_predictions"]
    save_dir = root_dir / "results"
    save = config.get("save", True)

    log_path = None
    if save:
        run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir / experiment_name / run_start_time
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "log.log"
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
    _logger = set_logger(
        name="matto_radiomics.poor_responder", level=logging.DEBUG, outpath=log_path
    )
    _logger.info("Starting Good/Poor responder stratification")

    os_predictions = pd.read_csv(os_predictions_path, index_col=0)
    ar_predictions = pd.read_csv(ar_predictions_path, index_col=0)

    combined = stratify_responders(os_predictions, ar_predictions)
    _logger.info("Combined stratification:\n%s", combined)

    if save:
        combined.to_csv(save_dir / "predictions.csv")
        _logger.info("Predictions saved to %s", save_dir / "predictions.csv")
