"""
Core simulation package.

The `engine` package is split by responsibility:

- `resources`: shared resource vocabulary and access profiles
- `agents`: mutable agent objects
- `state`: read-only public snapshots
- `actions`: small data objects exchanged between modules
- `build_rules`: build legality and build-side mutation
- `policies`: agent decision rules
- `institutions`: exchange mechanisms
- `game`: round orchestration and cross-agent mutation
"""
