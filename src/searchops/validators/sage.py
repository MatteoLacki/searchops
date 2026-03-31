import json


from pydantic import BaseModel, Field, validator
from typing import Dict, List, Tuple


# --- Enzyme ---------------------------------------------------------------


class Enzyme(BaseModel):
    missed_cleavages: int = 2
    cleave_at: str = "KR"
    restrict: str = "P"
    min_len: int = 7
    max_len: int = 50
    c_terminal: bool = True

    @validator("min_len")
    def check_lengths(cls, v, values):
        if "max_len" in values and v > values["max_len"]:
            raise ValueError("min_len cannot exceed max_len")
        return v


# --- Database -------------------------------------------------------------


class Database(BaseModel):
    bucket_size: int = 32768
    fragment_min_mz: float = 200.0
    fragment_max_mz: float = 1700.0
    peptide_min_mass: float = 600.0
    peptide_max_mass: float = 4000.0
    min_ion_index: int = 2

    enzyme: Enzyme = Field(default_factory=Enzyme)

    static_mods: Dict[str, float] = Field(default_factory=lambda: {"C": 57.0216})

    variable_mods: Dict[str, List[float]] = Field(
        default_factory=lambda: {"M": [15.9949], "[": [42.0]}
    )

    max_variable_mods: int = 2
    decoy_tag: str = "rev_"
    generate_decoys: bool = True

    @validator("fragment_max_mz")
    def check_fragment_range(cls, v, values):
        if "fragment_min_mz" in values and v <= values["fragment_min_mz"]:
            raise ValueError("fragment_max_mz must be > fragment_min_mz")
        return v

    @validator("peptide_max_mass")
    def check_peptide_range(cls, v, values):
        if "peptide_min_mass" in values and v <= values["peptide_min_mass"]:
            raise ValueError("peptide_max_mass must be > peptide_min_mass")
        return v


# --- Tolerances -----------------------------------------------------------


class Tolerance(BaseModel):
    ppm: Tuple[float, float]

    @validator("ppm")
    def check_ppm_range(cls, v):
        if v[0] >= v[1]:
            raise ValueError("ppm lower bound must be < upper bound")
        return v


# --- Root config ----------------------------------------------------------


class SageConfig(BaseModel):
    database: Database = Field(default_factory=Database)

    deisotope: bool = True
    chimera: bool = False
    wide_window: bool = False
    predict_rt: bool = True

    min_peaks: int = 10
    max_peaks: int = 800
    min_matched_peaks: int = 4

    max_fragment_charge: int = 1
    ignore_precursor_charge: bool = False

    parallel: bool = True
    report_psms: int = 1

    precursor_tol: Tolerance = Field(default_factory=lambda: Tolerance(ppm=(-12, 12)))
    fragment_tol: Tolerance = Field(default_factory=lambda: Tolerance(ppm=(-15, 15)))

    isotope_errors: Tuple[int, int] = (0, 3)

    @validator("max_peaks")
    def check_peaks(cls, v, values):
        if "min_peaks" in values and v < values["min_peaks"]:
            raise ValueError("max_peaks must be >= min_peaks")
        return v

    @validator("isotope_errors")
    def check_isotope_range(cls, v):
        if v[0] > v[1]:
            raise ValueError("invalid isotope error range")
        return v


def validate_config(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return SageConfig(**cfg)


if __name__ == "__main__":
    validated = validate_config(**cfg)

    print(validated.model_dump())
