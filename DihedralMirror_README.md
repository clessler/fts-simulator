# DihedralMirror Class

## Overview
The `DihedralMirror` class implements a V-shaped mirror with two flat surfaces meeting at an angle. Each surface is defined by the dihedral sag equation:

```
z = sign * (m * |y| + b)
```

where:
- `m` is the slope parameter (determines the angle between surfaces)
- `b` is the offset at y=0
- `sign` is +1 or -1 for the two surfaces

## Key Features

1. **Exact Intersection Finding**: Uses analytical solutions instead of numerical root-finding for fast, precise ray-surface intersections.

2. **Direction-Based Surface Selection**: Rays interact with only one surface based on their direction:
   - Rays with positive z-component (pointing "up") hit the negative surface (sign=-1)
   - Rays with negative z-component (pointing "down") hit the positive surface (sign=+1)

3. **Aperture Limiting**: The `diam` parameter limits the mirror size in the xy-plane (circular aperture).

4. **Dual Mode Support**:
   - **Sequential mode**: Works in simple list-based optical systems
   - **Branch/Graph mode**: Supports port-based routing with ports "pos" (positive surface), "neg" (negative surface), and "default" (missed rays)

## Class Definition

```python
@dataclass
class DihedralMirror(_PoseMixin, _RayOpsMixin):
    m: float          # slope parameter for dihedral
    b: float          # offset parameter for dihedral
    diam: float       # diameter to limit xy extent
    origin: tuple     # global position
    gx_local: tuple   # local x-axis direction
    gy_local: tuple   # local y-axis direction
    gz_local: tuple   # local z-axis direction
```

## Methods

### `trace(starting_rays, n_scan=2000, debug=False, root_finder=None, return_ports=False)`
Main ray tracing method. Returns traced rays, optionally with port information for branch routing.

### `plot(num_points=100, fig=None, opacity=0.8)`
Plots both surfaces of the dihedral mirror in 3D.

### `_find_intersection(ray, t_min=0.01, debug=False)`
Finds the intersection point of a ray with the appropriate dihedral surface.

## Usage Examples

### Sequential Mode

```python
import ray_tracing as sims
import numpy as np

# Create dihedral mirror
dihedral = sims.DihedralMirror(
    m=0.5, b=0.0, diam=50.0,
    origin=(0, 0, 0),
    gx_local=(1, 0, 0),
    gy_local=(0, 1, 0),
    gz_local=(0, 0, 1)
)

# Create optical system
optical_system = [dihedral]

# Create starting rays
starting_rays = [
    [0.0, 1.0, np.array([0.0, 10.0, 20.0]), np.array([0.0, 0.0, -1.0]), 0.0],
]

# Trace
result = sims.trace_optical_system(optical_system, starting_rays, plot=False)
```

### Graph Mode with Branch Routing

```python
optical_system_graph = {
    "entry": "dihedral",
    "nodes": {
        "dihedral": {
            "element": dihedral,
            "next": {
                "pos": "next_element_for_positive_surface",
                "neg": "next_element_for_negative_surface",
                "default": None  # Rays that miss
            }
        }
    }
}

result = sims.trace_optical_system(
    optical_system_graph,
    starting_rays,
    plot=False,
    return_all=True
)
```

## Implementation Details

### Analytical Intersection Finding

The intersection is found by solving:
```
z0 + t*dz = sign * (m * |y0 + t*dy| + b)
```

This splits into two cases based on the sign of (y0 + t*dy):
1. Case y ≥ 0: `t*(dz - sign*m*dy) = sign*m*y0 + sign*b - z0`
2. Case y < 0: `t*(dz + sign*m*dy) = -sign*m*y0 + sign*b - z0`

The code evaluates both cases and selects the valid solution.

### Surface Normal Calculation

For the dihedral surface z = sign * (m * |y| + b):
- The derivative dz/dy = sign * m * sign(y) for y ≠ 0
- The normal vector is computed from the cross product of tangent vectors
- At y=0 (the kink), a consistent normal is chosen based on the sign

## Files

- `ray_tracing.py`: Contains the DihedralMirror class (lines 639-888)
- `dihedral_mirror_example.py`: Complete usage examples
- `test_dihedral_mirror.py`: Comprehensive tests

## Testing

Run the example:
```bash
conda activate optics_env
python dihedral_mirror_example.py
```

This will demonstrate:
1. Sequential mode tracing
2. Graph mode with branch routing
3. Complex multi-element systems
