# Bench fixtures

Readings the bench calibration tool (cls-cal, A. Raggio) took on the CLS rack on 24 to 26 October 2026, and what cls-cal itself made of them. `tests/test_bench_data.py` runs ORACLE's analyses on the same readings and requires the same answers. It also checks the simulated rack against them, since that rack is built from these records.

| File | cls-cal run | What |
|---|---|---|
| `linearity_1kv.npz` | `20261024_184702_linearity` | ±1 kV, 50 V steps, fine grid across 0 V, up then down, 0 V every 10: every point |
| `linearity_3kv_25oct.npz` | `20261025_021414_linearity_3kvspan` | ±3 kV in 1 V steps, up then part of the way down (stopped): sweep points |
| `linearity_3kv_26oct.npz` | `20261026_111717_linearity` | the same sweep a day later, −3000 to −226 V: sweep points |
| `zero_3kv_25oct.npz` | `20261025_021414_linearity_3kvspan` | the 0 V checks between that sweep's points (every 10th), for the bench table's 0 V correction |
| `zero_3kv_26oct.npz` | `20261026_111717_linearity` | the same for the sweep a day later |
| `settle.npz` | `20261024_142142_settle_LCR` | 9 steps × 3 repeats, 0.5 ms samples at a 200 µs aperture, platform volts |
| `cooler.npz` | `20261024_141540_cooler` | 60 s of the cooler every 1 ms at a 200 µs aperture, cooler volts |
| `reference.json` | all of the above | cls-cal's own results and the settings it ran with |

Each `.npz` holds the columns its test reads, in volts at the platform or the cooler. `kind` is 0 for a sweep point and 1 for a 0 V check. `direction` is 0 for none, 1 up, 2 down and 3 random.

`extract.py` made them from `ORIGINAL_DAQ_STUFF/cls_calibration_tests/data`. Re-run it from the repository root only if those records change.
