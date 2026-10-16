# S1/S1a signal processing and vendor boundary

Use `latentgait extract-features` only on authorised local 100-Hz tri-axial acceleration. Sensor axes must be mapped to physical vertical, mediolateral and anteroposterior directions. A signal CSV has `time_s,v,ml,ap`; time must be uniformly spaced at .01 s. Absolute paths and confirmed full-stride sample boundaries are supplied in a separate local JSON manifest:

```json
{
  "records": [
    {
      "participant_id": "LOCAL_IDENTIFIER",
      "trial_id": "LOCAL_TRIAL_IDENTIFIER",
      "signal_csv": "/secure/location/trial.csv",
      "stride_boundaries": [0, 100, 200]
    }
  ]
}
```

The three shown boundaries are a schematic format example, not a valid trial: at least 24 full strides (25 boundaries) are needed to retain 20 after first/last two are removed. Boundaries refer to successive **same-foot full gait cycles**, not every alternating foot contact. The upper endpoint can equal signal length. The package does not infer subject identity or a valid trial from this example.

```sh
cp configs/signal_protocol.example.yaml /secure/location/signal_protocol.yaml
# Resolve all signal-processing choices in the secure copy.
latentgait extract-features --manifest /secure/location/trials.json \
  --protocol /secure/location/signal_protocol.yaml \
  --output /secure/location/custom_features.csv
```

The output is private: it contains linkage keys, retained stride count and the 12 custom trunk features. Combine it with clinical/QC/BTS data using `assemble`; the exporter does not manufacture pelvis or spatiotemporal quantities.

## Directly specified equations

HR and iHR use demeaned acceleration, fourth-order 20-Hz Butterworth filtering with zero-phase forward/backward application, and Fourier coefficients for harmonics 1..20 of each retained stride. HR divides intrinsic harmonic amplitude sum by extrinsic amplitude sum. iHR is 100 times intrinsic power sum divided by total power. Intrinsic harmonics are even for AP/V, odd for ML. HR and iHR are averaged across retained strides. The paper explicitly specifies this iHR mapping; no phase-mapping TODO remains.

RQA uses a 2,000-point RBF-resampled retained segment, time-delay embedding, AMI's first minimum over delays 5..19 with bins within 4..19, and FNN's smallest dimension 3..10 satisfying <=1% false neighbours for declared Rtol 15..19 and Atol 2..4. Recurrence radius is a declared .05..20 fraction of maximum phase-space distance. %DET sums diagonal recurrence points in lines of length >=lmin and divides by all diagonal recurrence points in the upper triangle, excluding the main diagonal (and any explicitly chosen additional Theiler exclusion).

MSE uses the central 20 retained strides and RBF resampling to 2,000 points. Normalize to mean zero/population SD one, average non-overlapping groups of four, and compute sample entropy at m=2, delay=1, tolerance .10 of the normalized **pre-coarse-graining** SD. Self-matches are excluded. Zero matches are undefined and raise a QC error; they are not replaced by arbitrary finite entropy values.

## Explicit numerical conventions and unavailable original details

S1a reports parameter ranges but does not give the per-record selected values or a unique rule for combining candidate histogram/tolerance/radius choices. The package requires a chosen bin count, FNN tolerance pair, radius fraction and lmin/Theiler setting. If AMI has no interior minimum or no FNN dimension meets the criterion, extraction stops. It does not select a default from an endpoint. It records algorithms in readable functions for comparison with the authors' original routines.

The RBF kernel, smoothing, neighbours and epsilon are explicit parameters because “RBF interpolation” alone does not identify them. The implementation uses SciPy RBFInterpolator. FNN/recurrence use Euclidean distance; sample entropy uses inclusive Chebyshev-distance matches with equal m/m+1 template support. AMI uses fixed whole-segment equal-width histogram edges and a first interior local minimum. Odd central-stride ties use the earlier central window. These are disclosed software conventions, not details claimed to have been specified by the final DOCX. Matching the original numerical feature values requires validation against the original custom signal routine or an author-approved parameter record.

`event_candidates` supplies a detrended 3.2-Hz FIR signal, positive-between-zero-crossings candidate peaks, and a gaus1 scale-9 CWT for local event review. FIR length/window, signal boundary treatment and the original wavelet event-selection procedure are under-specified. Consequently this helper is a QC aid; the production extractor requires externally verified full-stride boundaries. No automatic initial-contact detector is represented as a verified reconstruction of the unavailable original event code.

BTS G-Walk pelvis ranges and gait speed/cadence/stride-length/support variables require validated manufacturer exports, including software version, coordinate/range conventions and units. Trunk acceleration alone cannot uniquely reconstruct proprietary pelvic angle estimation or all manufacturer spatiotemporal algorithms. No replacement algorithm is passed off as BTS software.
