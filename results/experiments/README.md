# Development history

- `v1-newsgroups.json`: initial algorithm, before giant-cluster rejection and
  MiniBatchKMeans candidates. This failed to beat the upstream baseline.
- `v1-stress.json`: initial version's synthetic 5,000-text tests. Not final code.
- `interrupted-synthetic.json`: an early benchmark run where psutil could not
  inspect process IDs in the sandbox. Two candidate cases failed before running
  clustering; later workers used the corrected time.process_time measurement.
  This mixed development run is retained for transparency, not headline results.

Use the `*-final.json` files and `newsgroups-5000.json` in the parent directory
for the final candidate's measurements. None is an Apex live score.
