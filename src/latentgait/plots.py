"""Optional private figures computed from live fitted models, never paper constants."""

import numpy as np

from .models import posterior_values, probability_draws


def fitted_figures(output, grid, primary_logit, forms, spline, mean, sd, *, synthetic=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.special import ndtr

    prefix = "SYNTHETIC EXECUTION CHECK — " if synthetic else ""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(mean, bins=20)
    axes[0].set(xlabel="Latent posterior mean (standardized)", ylabel="Participants")
    axes[1].hist(sd, bins=20)
    axes[1].set(xlabel="Latent posterior SD (same standardized scale)", ylabel="Participants")
    fig.suptitle(prefix + "Measurement-model score distribution")
    fig.tight_layout()
    fig.savefig(output / "measurement_distributions.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    p = probability_draws(primary_logit, grid, "logit")
    lo, hi = np.quantile(p, [0.025, 0.975], axis=0)
    ax.plot(grid, p.mean(0))
    ax.fill_between(grid, lo, hi, alpha=0.2)
    ax.set(xlabel="Latent instability (SD)", ylabel="Fitted retrospective fall probability", ylim=(0, 1))
    ax.set_title("Primary logistic association\nAge/duration at cohort means")
    if synthetic:
        fig.suptitle("SYNTHETIC EXECUTION CHECK", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "primary_logistic_association.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for name, trace in forms.items():
        p = probability_draws(trace, grid, name, spline=spline)
        axes[0].plot(grid, p.mean(0), label=name)
        axes[1].plot(grid, np.gradient(p, grid, axis=1).mean(0), label=name)
    axes[0].set(ylabel="Fitted fall probability", ylim=(0, 1))
    axes[1].set(ylabel="dP/dZ")
    fixed = forms["fixed"]
    for shift in (0.1, 0.2, 0.3, 0.5):
        contrast = probability_draws(fixed, grid - shift, "fixed") - probability_draws(fixed, grid, "fixed")
        axes[2].plot(grid, contrast.mean(0), label=f"shift {shift}")
    axes[2].set(ylabel="Hypothetical change in fitted probability")
    for ax in axes:
        ax.set_xlabel("Latent instability (SD)")
        ax.legend(frameon=False)
    fig.suptitle(prefix + "Functional-form and hypothetical-contrast summaries")
    fig.tight_layout()
    fig.savefig(output / "functional_forms_and_contrasts.pdf")
    plt.close(fig)

    tau = posterior_values(fixed, "tau")[:, None]
    z = posterior_values(fixed, "z_true")
    delta = (ndtr(z - 0.3 - tau) - ndtr(z - tau)).mean(0)
    order = np.argsort(mean)
    weights = abs(delta[order])
    if weights.sum() > 1e-15:
        from .reporting import cumulative_contrast_interval

        central = cumulative_contrast_interval(mean, abs(delta), rule="central_90")["interval"]
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.plot(mean[order], np.cumsum(weights) / weights.sum())
        ax.axvspan(central[0], central[1], alpha=0.15, label="Central 90% absolute contrast")
        ax.set(xlabel="Baseline latent instability (SD)", ylabel="Cumulative absolute fitted contrast", ylim=(0, 1))
        ax.set_title("Descriptive cumulative\n0.3-SD hypothetical contrast")
        ax.legend(frameon=False)
        if synthetic:
            fig.suptitle("SYNTHETIC EXECUTION CHECK", fontsize=10)
        fig.tight_layout()
        fig.savefig(output / "cumulative_hypothetical_contrast.pdf")
        plt.close(fig)
