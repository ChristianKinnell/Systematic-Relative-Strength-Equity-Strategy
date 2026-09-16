# Reproducibility

## Frozen research artefacts
A reproducible run should preserve:
- exact source commit/tag;
- `StrategyConfig`;
- candidate-universe tickers, sectors and cost assumptions;
- price parquet;
- snapshot manifest including download timestamp, date range and package versions;
- generated diagnostics and headline metrics.

The model supports:

```bash
NYSE_ALPHA_FREEZE_SNAPSHOT=1 python scripts/run_frozen.py
NYSE_ALPHA_USE_SNAPSHOT=1 python scripts/run_frozen.py
```

`prices.parquet` is ignored by Git because it can be large and may be subject to data-provider redistribution constraints. `universe.csv` and `manifest.json` can be retained locally or attached to a research release if appropriate.

## Research fingerprint
`research_snapshot/manifest.json` contains the SHA-256 hash of the frozen v5 monolith and its frozen configuration. A change to strategy code should create a new version rather than overwrite the v5 snapshot.

## Refactor parity gate
The archive monolith is canonical until modular execution matches, at minimum:
1. daily target weights;
2. daily net return and equity;
3. rebalance dates;
4. circuit-breaker flags;
5. transaction costs;
6. final target portfolio;
7. key validation outputs within numerical tolerance.
