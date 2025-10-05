"""Exoplanet synthetic data generation utilities.

This module provides a reproducible pipeline that learns the joint distribution of the
feature engineered Kepler dataset and allows sampling new, physically plausible
candidates for advanced experimentation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# Feature selection inspired by the training pipeline in ``main.py``
IMPORTANT_FEATURES: Tuple[str, ...] = (
    "koi_period",
    "koi_depth",
    "koi_duration",
    "koi_ror",
    "koi_srho",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_impact",
    "koi_model_snr",
    "koi_tce_plnt_num",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
)

# Conservative physical bounds based on the Kepler cumulative catalogue
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "koi_period": (0.05, 1000.0),
    "koi_depth": (1.0, 20000.0),
    "koi_duration": (0.1, 50.0),
    "koi_ror": (1e-4, 0.5),
    "koi_srho": (0.01, 30.0),
    "koi_prad": (0.1, 50.0),
    "koi_teq": (50.0, 4000.0),
    "koi_insol": (0.001, 1e4),
    "koi_steff": (2500.0, 10000.0),
    "koi_slogg": (2.5, 5.5),
    "koi_srad": (0.01, 30.0),
    "koi_impact": (0.0, 1.5),
    "koi_model_snr": (0.0, 500.0),
    "koi_tce_plnt_num": (1.0, 7.0),
    "koi_fpflag_nt": (0, 1),
    "koi_fpflag_ss": (0, 1),
    "koi_fpflag_co": (0, 1),
    "koi_fpflag_ec": (0, 1),
}

DERIVED_FEATURES: Tuple[str, ...] = (
    "planet_star_ratio",
    "signal_quality",
    "orbital_velocity",
    "fp_total_score",
    "transit_shape_factor",
)


@dataclass
class SyntheticDataset:
    """Container for generated samples."""

    features: pd.DataFrame
    labels: pd.Series

    def to_dataframe(self, include_labels: bool = True) -> pd.DataFrame:
        df = self.features.copy()
        if include_labels:
            df["is_exoplanet"] = self.labels.values
        return df


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the feature engineering logic used during training."""

    # Ensure required columns exist before feature engineering
    engineered = df.copy()

    if {"koi_prad", "koi_srad"}.issubset(engineered.columns):
        engineered["planet_star_ratio"] = engineered["koi_prad"] / (
            engineered["koi_srad"] * 109.2
        )

    if {"koi_depth", "koi_model_snr"}.issubset(engineered.columns):
        engineered["signal_quality"] = (
            engineered["koi_depth"] * engineered["koi_model_snr"]
        )

    if "koi_period" in engineered.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            engineered["orbital_velocity"] = 1.0 / engineered["koi_period"].replace(
                0, np.nan
            )

    fp_flags = [
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
    ]
    available_flags = [flag for flag in fp_flags if flag in engineered.columns]
    if available_flags:
        engineered["fp_total_score"] = engineered[available_flags].sum(axis=1)

    if {"koi_duration", "koi_period"}.issubset(engineered.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            engineered["transit_shape_factor"] = engineered["koi_duration"] / (
                engineered["koi_period"].replace(0, np.nan)
            )

    engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    engineered.fillna(engineered.median(numeric_only=True), inplace=True)

    feature_cols = [
        col
        for col in list(IMPORTANT_FEATURES) + list(DERIVED_FEATURES)
        if col in engineered.columns
    ]

    return engineered[feature_cols].copy()


class ExoplanetDataGenerator:
    """Model-based synthetic data generator for the Kepler exoplanet problem."""

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "full",
        random_state: int = 42,
        smote_k_neighbors: int = 5,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.smote_k_neighbors = smote_k_neighbors

        self.scaler = RobustScaler()
        self.class_models: Dict[int, GaussianMixture] = {}
        self.feature_names: Optional[pd.Index] = None
        self.feature_quantiles: Optional[pd.DataFrame] = None
        self.class_priors: Dict[int, float] = {}
        self._fitted_data: Optional[pd.DataFrame] = None
        self._fitted_labels: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExoplanetDataGenerator":
        """Fit Gaussian mixture models per class."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with named columns")

        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="is_exoplanet")

        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median(numeric_only=True))
        self.feature_names = X_clean.columns

        X_scaled = self.scaler.fit_transform(X_clean)
        self.feature_quantiles = X_clean.quantile([0.01, 0.05, 0.95, 0.99])

        self.class_models = {}
        self.class_priors = {}
        rng = np.random.default_rng(self.random_state)

        for cls in sorted(np.unique(y)):
            class_mask = y == cls
            X_cls = X_scaled[class_mask]
            if X_cls.shape[0] < self.n_components:
                n_components = max(1, X_cls.shape[0])
            else:
                n_components = self.n_components

            model = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                reg_covar=1e-3,
                init_params="kmeans",
                max_iter=400,
            )
            model.fit(X_cls)
            # Warm start by sampling to remove singularities
            model.sample(min(10, X_cls.shape[0]))
            self.class_models[int(cls)] = model
            self.class_priors[int(cls)] = float(class_mask.mean())

        # Persist fitted data for SMOTE-based strategies
        self._fitted_data = pd.DataFrame(X_scaled, columns=self.feature_names)
        self._fitted_labels = y.reset_index(drop=True)

        # Store RNG state for deterministic sampling
        self._rng = rng
        return self

    @staticmethod
    def _resolve_class_counts(
        total_samples: int, class_ratio: Optional[Dict[int, float]], priors: Dict[int, float]
    ) -> Dict[int, int]:
        if class_ratio:
            ratio_sum = sum(class_ratio.values())
            return {
                cls: int(np.round(total_samples * class_ratio.get(cls, 0.0) / ratio_sum))
                for cls in priors
            }
        return {
            cls: int(np.round(total_samples * priors.get(cls, 1.0 / len(priors))))
            for cls in priors
        }

    def _postprocess(self, samples: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(samples, columns=self.feature_names)

        # Bring values back to original space
        df = pd.DataFrame(
            self.scaler.inverse_transform(df.values), columns=self.feature_names
        )

        # Clip with empirical quantiles & hard physical bounds
        if self.feature_quantiles is not None:
            lower = self.feature_quantiles.loc[0.01]
            upper = self.feature_quantiles.loc[0.99]
            df = df.clip(lower=lower, upper=upper, axis=1)

        for feature, bounds in PHYSICAL_BOUNDS.items():
            if feature in df.columns:
                df[feature] = df[feature].clip(bounds[0], bounds[1])

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        return df

    def _sample_gaussian(
        self,
        n_samples: int,
        class_ratio: Optional[Dict[int, float]] = None,
    ) -> SyntheticDataset:
        if not self.class_models:
            raise RuntimeError("Generator must be fitted before calling sample().")

        class_counts = self._resolve_class_counts(n_samples, class_ratio, self.class_priors)
        generated_frames = []
        generated_labels = []

        for cls, count in class_counts.items():
            if count <= 0:
                continue
            model = self.class_models[int(cls)]
            samples_scaled, _ = model.sample(count)
            samples = self._postprocess(samples_scaled)
            generated_frames.append(samples)
            generated_labels.append(pd.Series([cls] * len(samples)))

        features = pd.concat(generated_frames, ignore_index=True)
        labels = pd.concat(generated_labels, ignore_index=True)
        return SyntheticDataset(features, labels)

    def _sample_smote(
        self,
        n_samples: int,
        class_ratio: Optional[Dict[int, float]] = None,
    ) -> SyntheticDataset:
        if self._fitted_data is None or self._fitted_labels is None:
            raise RuntimeError("Generator must be fitted before calling sample().")

        if n_samples <= 0:
            empty_features = pd.DataFrame(columns=self.feature_names)
            empty_labels = pd.Series(name="is_exoplanet", dtype=int)
            return SyntheticDataset(empty_features, empty_labels)

        class_counts = self._resolve_class_counts(n_samples, class_ratio, self.class_priors)
        minority_class = min(self.class_priors, key=self.class_priors.get)
        majority_class = max(self.class_priors, key=self.class_priors.get)

        desired_minority = class_counts.get(minority_class, 0)
        desired_majority = n_samples - desired_minority

        current_minority = int((self._fitted_labels == minority_class).sum())
        target_total_minority = current_minority + max(desired_minority, 1)

        smote = SMOTE(
            sampling_strategy={minority_class: target_total_minority},
            random_state=self.random_state,
            k_neighbors=min(self.smote_k_neighbors, max(1, current_minority - 1)),
        )

        X_res, y_res = smote.fit_resample(self._fitted_data, self._fitted_labels)
        new_mask = np.arange(len(X_res)) >= len(self._fitted_data)
        new_minority_scaled = X_res[new_mask][y_res[new_mask] == minority_class]
        minority_samples = self._postprocess(new_minority_scaled)[:desired_minority]

        majority_pool_scaled = self._fitted_data[self._fitted_labels == majority_class].values
        majority_pool = self._postprocess(majority_pool_scaled)
        rng = np.random.default_rng(self.random_state)
        majority_indices = rng.choice(len(majority_pool), size=max(desired_majority, 0), replace=True)
        majority_samples = majority_pool.iloc[majority_indices].reset_index(drop=True)

        features = pd.concat([majority_samples, minority_samples], ignore_index=True)
        labels = pd.Series(
            [majority_class] * len(majority_samples) + [minority_class] * len(minority_samples),
            name="is_exoplanet",
        )

        return SyntheticDataset(features, labels)

    def sample(
        self,
        n_samples: int,
        strategy: str = "gaussian",
        class_ratio: Optional[Dict[int, float]] = None,
        hybrid_split: float = 0.6,
    ) -> SyntheticDataset:
        """Generate synthetic samples using the selected strategy."""

        strategy = strategy.lower()
        if strategy not in {"gaussian", "smote", "hybrid"}:
            raise ValueError("strategy must be one of {'gaussian', 'smote', 'hybrid'}")

        if strategy == "gaussian":
            return self._sample_gaussian(n_samples, class_ratio)

        if strategy == "smote":
            return self._sample_smote(n_samples, class_ratio)

        # Hybrid: blend Gaussian mixture and SMOTE-based generation
        gaussian_count = int(n_samples * hybrid_split)
        smote_count = max(0, n_samples - gaussian_count)

        gaussian_data = self._sample_gaussian(gaussian_count, class_ratio)
        smote_data = self._sample_smote(smote_count, class_ratio)

        features = pd.concat(
            [gaussian_data.features, smote_data.features], ignore_index=True
        )
        labels = pd.concat(
            [gaussian_data.labels, smote_data.labels], ignore_index=True
        )
        return SyntheticDataset(features, labels)

    def generate_dataset(
        self,
        n_samples: int,
        strategy: str = "gaussian",
        class_ratio: Optional[Dict[int, float]] = None,
        include_labels: bool = True,
    ) -> pd.DataFrame:
        dataset = self.sample(n_samples, strategy=strategy, class_ratio=class_ratio)
        return dataset.to_dataframe(include_labels=include_labels)


def load_reference_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")
    df_raw = pd.read_csv(data_path, comment="#")
    if "koi_disposition" not in df_raw.columns:
        raise ValueError("Veri seti 'koi_disposition' sütununu içermeli.")

    df_raw["is_exoplanet"] = (
        df_raw["koi_disposition"].str.upper() == "CONFIRMED"
    ).astype(int)
    features = _feature_engineering(df_raw)
    return pd.concat([features, df_raw["is_exoplanet"]], axis=1)


def prepare_features_from_dataframe(
    df_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    if "is_exoplanet" not in df_raw.columns:
        if "koi_disposition" not in df_raw.columns:
            raise ValueError(
                "DataFrame içerisinde 'is_exoplanet' veya 'koi_disposition' sütunu bulunmalı."
            )
        df_raw = df_raw.copy()
        df_raw["is_exoplanet"] = (
            df_raw["koi_disposition"].str.upper() == "CONFIRMED"
        ).astype(int)

    features = _feature_engineering(df_raw)
    labels = df_raw["is_exoplanet"].astype(int)
    return features, labels


def generate_synthetic_dataset(
    data_path: Path,
    samples: int,
    strategy: str = "gaussian",
    output_path: Optional[Path] = None,
    class_ratio: Optional[Dict[int, float]] = None,
    include_labels: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    reference = load_reference_dataset(data_path)
    X = reference.drop(columns="is_exoplanet")
    y = reference["is_exoplanet"]

    generator = ExoplanetDataGenerator(random_state=random_state)
    generator.fit(X, y)
    synthetic_df = generator.generate_dataset(
        n_samples=samples,
        strategy=strategy,
        class_ratio=class_ratio,
        include_labels=include_labels,
    )

    if output_path:
        synthetic_df.to_csv(output_path, index=False)

    return synthetic_df


def _parse_ratio(values: Iterable[str]) -> Dict[int, float]:
    ratio = {}
    for value in values:
        key, raw = value.split("=")
        ratio[int(key)] = float(raw)
    return ratio


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Kepler exoplanet samples for experimentation.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("cumulative_2025.10.04_09.55.40.csv"),
        help="Referans Kepler CSV dosyasının yolu.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Üretilecek sentetik örnek sayısı.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="gaussian",
        choices=["gaussian", "smote", "hybrid"],
        help="Sentetik veri üretim stratejisi.",
    )
    parser.add_argument(
        "--class-ratio",
        type=str,
        nargs="*",
        help="Sınıf oranları (örn: 0=0.6 1=0.4). Toplam 1'e normalleştirilir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_exoplanets.csv"),
        help="Oluşturulacak CSV dosyasının yolu.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Çıktıya is_exoplanet etiketini ekleme.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="İlk 5 satırı terminale yazdır.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Rastgelelik kontrolü için tohum değeri.",
    )

    args = parser.parse_args()
    ratio = _parse_ratio(args.class_ratio) if args.class_ratio else None

    synthetic_df = generate_synthetic_dataset(
        data_path=args.data,
        samples=args.samples,
        strategy=args.strategy,
        class_ratio=ratio,
        output_path=args.output,
        include_labels=not args.no_labels,
        random_state=args.random_state,
    )

    if args.preview:
        print("\n📊 Üretilen örneklerden ilk 5 satır:\n")
        print(synthetic_df.head())
        print("\n✅ Toplam örnek sayısı:", len(synthetic_df))


if __name__ == "__main__":
    main_cli()
