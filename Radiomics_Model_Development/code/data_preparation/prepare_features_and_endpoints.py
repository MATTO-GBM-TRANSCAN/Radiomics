import argparse
from pathlib import Path

import pandas as pd


OUTPUT_ENDPOINT_COLUMNS = [
    "Patient",
    "Progression",
    "Time to Progression (TTP)",
    "Acute Recurrence",
    "Overall survival",
    "Death",
]


LEGACY_ENDPOINT_COLUMNS = {
    "Recurrence": "Progression",
    "Time_To_Progression": "Time to Progression (TTP)",
    "OS": "Overall survival",
    "Dead": "Death",
}


def _normalize_endpoints(endpoints: pd.DataFrame, endpoints_path: str) -> pd.DataFrame:
    endpoints = endpoints.copy()
    endpoints.columns = endpoints.columns.str.strip()
    endpoints = endpoints.rename(columns=LEGACY_ENDPOINT_COLUMNS)

    required_without_acute = [
        "Patient",
        "Progression",
        "Time to Progression (TTP)",
        "Overall survival",
        "Death",
    ]
    missing = [column for column in required_without_acute if column not in endpoints.columns]
    if missing:
        raise ValueError(
            f"{endpoints_path} is missing required endpoint column(s): {missing}. "
            "Use the current template columns or the legacy columns "
            "Recurrence, Time_To_Progression, OS and Dead."
        )

    computed_acute_recurrence = (
        endpoints["Progression"].astype(float)
        * (endpoints["Time to Progression (TTP)"].astype(float) <= 90)
    ).astype(int)
    if "Acute Recurrence" not in endpoints.columns:
        endpoints["Acute Recurrence"] = computed_acute_recurrence
    else:
        acute_recurrence = endpoints["Acute Recurrence"].replace("", pd.NA)
        endpoints["Acute Recurrence"] = acute_recurrence.fillna(
            computed_acute_recurrence
        ).astype(int)

    return endpoints[OUTPUT_ENDPOINT_COLUMNS]


def main(features_path: str, endpoints_path: str) -> None:
    features = pd.read_csv(features_path)
    features.columns = features.columns.str.strip()
    features.dropna(axis=0, inplace=True)

    if features["Patient"].duplicated().any():
        if "Contour" not in features.columns:
            duplicated_patients = sorted(
                features.loc[features["Patient"].duplicated(), "Patient"].unique()
            )
            raise ValueError(
                f"Patient(s) {duplicated_patients} appear more than once in "
                f"{features_path}, but there's no 'Contour' column to tell those rows "
                "apart. Add a 'Contour' column to input.csv (e.g. 'MR', 'PET') so each "
                "patient's rows can be combined into one, or extract each contour "
                "separately if you don't want them combined."
            )
        numeric = pd.concat(
            [features[["Patient", "Contour"]], features.select_dtypes("number").drop(columns=["Patient"], errors="ignore")],
            axis=1,
        )
        features = numeric.pivot(index="Patient", columns="Contour")
        features.columns = [f"{column}-{contour}" for column, contour in features.columns]
        features = features.reset_index()
    else:
        numeric_features = features.select_dtypes("number").drop(columns=["Patient"], errors="ignore")
        features = pd.merge(
            features["Patient"],
            numeric_features,
            left_index=True,
            right_index=True,
        )
    features.to_csv(
        Path(features_path).parent / "features.csv",
        index=False,
    )

    if endpoints_path is None:
        return
    endpoints = pd.read_csv(endpoints_path)
    endpoints = _normalize_endpoints(endpoints, endpoints_path)
    endpoints.to_csv(
        Path(endpoints_path).parent / "endpoints.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare radiomics features and endpoints."
    )
    parser.add_argument(
        "--features_path",
        required=True,
        help="Path to the radiomics features CSV file.",
    )
    parser.add_argument("--endpoints_path", help="Path to the endpoints CSV file.")
    args = parser.parse_args()

    main(args.features_path, args.endpoints_path)
