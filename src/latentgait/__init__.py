"""LatentGait: analysis code only; it ships no participant data or results."""

from .cohort import lock_subject_cohort
from .pca import fit_domain_pcas

__all__ = ["lock_subject_cohort", "fit_domain_pcas"]
