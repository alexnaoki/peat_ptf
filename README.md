# peat_ptf

Pedotransfer functions (PTFs) for estimating **Mualem–van Genuchten (MVG)** hydraulic parameters of peat soils.

The module implements the regression equations from **Table 4** and the representative grouped parameter sets from **Table 5** of the source publication, covering Sphagnum, Woody, and Sedge peat types across a range of bulk densities.

---

## Hydraulic Parameters

| Symbol | Description | Unit |
|--------|-------------|------|
| θs | Saturated water content | cm³ cm⁻³ |
| α | van Genuchten alpha | cm⁻¹ |
| n | van Genuchten n | — |
| Ks | Saturated hydraulic conductivity | cm h⁻¹ |
| τ | Mualem tortuosity | — |

## Input Variables

| Symbol | Description | Unit |
|--------|-------------|------|
| BD | Bulk density | g cm⁻³ |
| OM | Organic matter content | wt% |
| depth | Sample depth | cm |

---

## Installation

No external dependencies are required — the module uses only the Python standard library.

Simply clone the repository and import from `src/`:

```bash
git clone <repo-url>
cd peat_ptf
```

```python
import sys
sys.path.insert(0, "src")
from peat_ptf import get_mvg_parameters, ptf_sphagnum, MVGParameters
```

---

## Quick Start

### Using the convenience dispatcher

`get_mvg_parameters()` is the main entry point. It selects the appropriate equation set based on peat type, bulk density, and the chosen method.

```python
from peat_ptf import get_mvg_parameters

# Table 4 regression PTF for Sphagnum peat
params = get_mvg_parameters(
    BD=0.05,
    depth=10,
    OM=95,
    peat_type="sphagnum",
    method="ptf",
)
print(params)
# MVGParameters(θs=0.9281, α=0.3486, n=1.4006, Ks=10.9187, τ=-1.6920)
```

### Calling type-specific PTFs directly (Table 4)

```python
from peat_ptf import ptf_sphagnum, ptf_woody, ptf_sedge, ptf_all_types_high_bd

# Sphagnum (BD ≤ 0.2)
params = ptf_sphagnum(BD=0.05, OM=95, depth=10)

# Woody (BD ≤ 0.2)
params = ptf_woody(BD=0.10, depth=20, OM=90)

# Sedge (BD ≤ 0.2)
params = ptf_sedge(BD=0.15, depth=30)

# All types, high bulk density (BD > 0.2)
params = ptf_all_types_high_bd(BD=0.35)
```

### Using grouped parameter look-ups (Table 5)

These return fixed representative parameter sets for bulk-density subgroups.

```python
from peat_ptf import get_mvg_parameters

# MVG-optimised grouped values for Woody peat
params = get_mvg_parameters(BD=0.08, peat_type="woody", method="grouped_mvg")
print(params)
# MVGParameters(θs=0.9600, α=21.3800, n=1.2600, Ks=40.1200, τ=-4.6900)

# PTF-derived grouped values for Sphagnum peat
params = get_mvg_parameters(BD=0.03, peat_type="sphagnum", method="grouped_ptf")
print(params)
# MVGParameters(θs=0.9400, α=0.7840, n=1.4500, Ks=58.4100, τ=-2.9000)
```

### Accessing individual parameters

The returned `MVGParameters` object is a dataclass, so fields are accessed directly:

```python
params = get_mvg_parameters(BD=0.10, depth=15)
print(f"Ks = {params.Ks:.2f} cm/h")
print(f"α  = {params.alpha:.4f} cm⁻¹")
print(f"n  = {params.n:.4f}")
```

---

## Available Methods

| `method` | Source | Description |
|----------|--------|-------------|
| `"ptf"` | Table 4 | Regression equations — continuous prediction from BD, OM, depth |
| `"grouped_mvg"` | Table 5 | MVG-optimised representative values per BD subgroup |
| `"grouped_ptf"` | Table 5 | PTF-derived representative values per BD subgroup |

## Peat Types and BD Ranges

| Peat type | BD range (Table 4 PTF) | BD groups (Table 5) |
|-----------|------------------------|---------------------|
| Sphagnum | ≤ 0.2 | I (≤0.05), III (0.05–0.1), IV (0.1–0.2), V (>0.2) |
| Woody | ≤ 0.2 | I (≤0.1), III (0.1–0.2), IV (>0.2) |
| Sedge | ≤ 0.2 | I (≤0.1), II (0.1–0.2) |
| All types | > 0.2 | I (>0.2) |

---

## Project Structure

```
peat_ptf/
├── README.md
├── src/
│   └── peat_ptf.py      # All PTF functions and MVG parameter dataclass
└── notebooks/
    └── peat_ptf.ipynb    # Interactive examples and visualisations
```

---

## License

See the repository for licence details.
