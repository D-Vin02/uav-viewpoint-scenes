# Multi-UAV Exploration Scenarios

This repository accompanies *A Quality-Aware Cooperative Method for Multi-UAV Exploration under Operational Constraints*. It contains the synthetic scenario generator, the publication configuration, and the original scenario splits used in the paper's synthetic experiments.

| Split | Scenarios | Use |
| --- | ---: | --- |
| [Training](scenarios/train) | 100 | Policy training |
| [Validation](scenarios/validation) | 20 | Checkpoint selection |
| [Test](scenarios/test) | 30 | Evaluation of frozen policies |

Each JSON scene contains 32 viewpoint positions, 80 target sample positions, the undirected flight edges, and the base viewpoint index. Indices are zero-based. File names use split-local indices. [The manifest](scenarios/manifest.json) records the actual random seed, difficulty profile, complete generation parameters, graph statistics, and SHA-256 checksum for each scene.

## Regenerate the publication splits

Use Python 3.12 and run the following commands from the repository root. The output directory must be new.

```bash
python -m pip install -r requirements.txt
python generate_scenarios.py --config configs/v2_team4.json --output-dir regenerated
```

The configuration uses master seed `20260914` and requires at least four edges incident to the base in each scene. Training, validation, and test use distinct seed ranges. The generator writes the archive's original CRLF line endings to preserve SHA-256 checksums across platforms.

The synthetic generator has no external data input. For the separate real-world experiments, the source dataset is available from the [UrbanBIS project](https://vcc.tech/UrbanBIS).
