# Tests

This directory contains tests organized by their runtime requirements.

## Directory Structure

```
tests/
├── local/                      # Tests that run on any machine (no cluster access)
│   ├── test_heartbeat.py       # Unit tests for HeartbeatWriter
│   └── test_imports.sh         # Smoke test for imports
│
└── cluster/                    # Tests requiring cluster/scheduler access
    └── lsf/                    # LSF-specific tests
        └── test_heartbeat_lsf_e2e.py   # End-to-end heartbeat test on LSF
```

## Running Tests

### Local Tests (no cluster required)

Run from repo root:

```bash
# Unit tests for heartbeat
make test-heartbeat

# Import smoke tests
make test-sanity
```

Or directly:

```bash
cd tdmpc2
python -m unittest tests.local.test_heartbeat -v
./tests/local/test_imports.sh
```

### Cluster Tests (LSF required)

Submit to LSF cluster:

```bash
make test-heartbeat-lsf
```

Or manually:

```bash
cd tdmpc2
bsub < jobs/test_heartbeat_e2e.lsf
```

## Conventions

- **`local/`** – Tests here must run without GPU, cluster access, or heavy dependencies.
- **`cluster/<scheduler>/`** – Tests here require the named scheduler (e.g., `lsf/`).
- **Naming** – Cluster tests include the scheduler in the filename (e.g., `*_lsf_*.py`).
- **Never auto-run cluster tests** – They should only be invoked via explicit `make test-*-lsf` or `bsub` commands.

