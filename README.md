# Route Resonators

Route Resonators generates paths for open-closed resonant cavities inside closed geometries. The main branch supports most arbitrary geometries, while the `nit_sample` branch is specifically intended for rectangular samples for the normal incidence impedance tube. The routing algorithm is the same regardless of the sample geometry, however, the setup specifically the distribution of the cavity openings on the surface of the geometries differs. 

The routing algorithm first discretizes the geometry on a Cartesian grid, determines the distribution of the resonators across the surface, and then routes each cavity through the interior until the length requirement is met. It also tracks previously routed cavities to prevent overlap. 

## Installation and Dependencies

This repository is a lightweight Python utility set rather than a packaged pip library. The primary dependencies are:

- h5py
- numpy
- scipy
- pyvista (https://github.com/pyvista/pyvista)
- aerosandbox (https://github.com/peterdsharpe/AeroSandbox)

All the dependencies can be installed as follows
```bash
pip install -r requirements.txt
```

## Quick Start

Run the default example from the repository root:

```bash
python route_resonator.py
```

The script reads `input_params.json5`, routes the requested resonators, prints progress and summary metrics, and creates a `res_paths/` directory containing files such as:

```text
res_paths/
  res0.csv
  res1.csv
  res2.csv
  ...
```

Each CSV contains one 3D point per row in `x,y,z` order, in the same length units used by the input parameters. The default parameters use metres, so the values are written in metres.

The output directory is deleted and recreated on every run. Move or rename any results that need to be retained before starting another run.

## Command-Line Usage

```text
python route_resonator.py [-h] [-input_params FILE] [-stl_file FILE]
```

Options:

- `-input_params FILE`: JSON5 parameter file. Defaults to `input_params.json5`.
- `-stl_file FILE`: optional STL filename. It overrides `STL_file` in the parameter file, but STL geometry is not loaded on the `nit_sample` branch. The dimensions of the NIT sample are set directly in the `input_params.json5` file.
- `-h`, `--help`: displays the arguments provided by `argparse`.

Examples:

```bash
python route_resonator.py -input_params input_params.json5
python route_resonator.py -input_params my_sample.json5 -stl_file my_sample.stl
```

The second command is accepted by the CLI, but only on the main branch. If using the `nit_sample` branch the sample dimensions are defined in the `input_params.json5` file.

## Input Parameters

Parameters are read as JSON5, so comments are allowed. The included example is:

| Parameter | Type | Meaning |
| --- | --- | --- |
| `STL_file` | string | STL filename stored in the input configuration. Required unless `-stl_file` is provided; currently not used to construct the geometry. |
| `a` | number | Resonator radius, in metres in the example. It controls grid spacing through `dx = 2.5 * a`. |
| `L` | number array | Requested resonator lengths. A single value applies to all resonators; multiple values are assigned according to the generated resonator types. |
| `r_bend` | number | Requested bend radius. The effective radius is limited to the grid spacing. |
| `dimensions` | six-number array | Box bounds in the order `[x_min, x_max, y_min, y_max, z_min, z_max]`. |
| `border` | number | Border width around the perimeter of the top surface. It is compared with the shell thickness before placement. |
| `shell_thickness` | number | Inward offset used to define the usable interior of the box. |
| `OAR` | number | Target open-area ratio used to estimate how many resonators to place. For example, `0.15` means 15%. |
| `uniform` | boolean | Selects uniform surface placement when true. The current entry point always calls the placement function with uniform routing enabled, while the function also reads this input value. |
| `pre_route` | boolean | When true, reserves an initial vertical section for each resonator before the main routing stage. This can help thin geometries clear space for bends. |
| `truncated` | boolean | When true, writes only endpoints and bend-related points instead of every grid point along a straight segment. |

### Parameter Notes

- `dimensions`, `a`, `L`, `r_bend`, `border`, and `shell_thickness` must use compatible units.
- The requested number of resonators is estimated from `OAR` and the top-surface area, then limited by the available grid and border.
- If the requested packing is too dense or the requested lengths cannot fit, some resonators may fail to route. This is reported as the success rate and is not raised as an exception.
- `L` should contain positive values. Very long cavities, large radii, a large shell thickness, or a large border can leave too little free volume for routing.

## Processing Pipeline

The entry point performs these stages:

1. Parse the JSON5 input and command-line overrides.
2. Construct a PyVista box and an inward-offset surface using `shell_thickness`.
3. Create a Cartesian grid around the geometry and classify usable cells with implicit-distance calculations.
4. Select resonator openings on the top surface, honoring the border and target open-area ratio.
5. Reserve starting nodes and assign requested lengths to resonator types.
6. Route resonators through unoccupied neighboring grid cells, preferring the current direction and otherwise favoring longer available runs.
7. Backtrack at dead ends and release cells that are no longer part of a candidate path.
8. Adjust bend points, optionally truncate straight runs, print metrics, and write CSV files.

## Console Metrics

Typical runs report:

- `Maximum OAR`: the largest open-area ratio permitted by the available top-surface grid.
- `Actual OAR`: the ratio after the discrete uniform placement has been selected.
- `Volume ratio`: routed resonator volume divided by the offset geometry volume, printed as a percentage.
- `Success rate`: percentage of initially placed resonators that produced a valid path.
- `Packing Efficiency`: percentage of initially unoccupied grid nodes used by the returned paths.

Successful routes also print routing progress and `Path found!` messages while the algorithm is running.

## Output Contract

`write_res_paths()` always writes to `./res_paths` relative to the current working directory. Existing contents are removed first. For each non-empty route, `res{i}.csv` contains an array of coordinates with no header:

```text
x0,y0,z0
x1,y1,z1
x2,y2,z2
```

When `truncated` is false, the file contains the retained grid points along the complete route. When it is true, intermediate points along straight sections are omitted, leaving a compact polyline around endpoints and bends.

## Routing Algorithm

<img width="2600" height="2600" alt="routing_flow_chart" src="/routing_flow_chart.png" />

The first step of the algorithm is to embed the geometry within a Cartesian grid, where the grid spacing is determined by the resonator radius plus an additional tolerance to account for clearance between adjacent cavities. This introduces a key limitation of the method, namely that all cavities must have identical cross-sectional dimensions. The surface of the geometry is then interpolated on the grid using a multidimensional piecewise-cubic function. 

Next, grid points lying outside the geometry are identified and marked as occupied. There are numerous methods that may be employed for this purpose. For instance, the signed distance function (SDF) representation of the geometry can be evaluated across the grid using methods, such as the Fast Marching Method (FMM) or Fast Sweeping Method (FSM). With this functional representation of the geometry, the grid points that lie within the geometry have negative SDF values, while those that fall outside of it have positive values. This approach is particularly useful for complex geometries that may have overlapping surfaces.  

Following this, the starting nodes of all the resonators on the surface of the geometry are specified. The points can be distributed uniformly along the primary directions or randomly but evenly using Poisson Disc Sampling. It is often advantageous to pre-route the resonators by advancing their starting nodes slightly into the interior. This ensures that each resonator can enter the geometry without being inadvertently blocked by previously routed cavities.

Once these positions are defined, each resonator is routed through the geometry sequentially. The routing order may be random or specified based on the geometry. For the rod samples considered for the experiment, a random routing order was found to allow for a greater number of resonators to fit inside them.

The routing procedure begins by examining all neighboring grid points along each coordinate direction surrounding the starting node. The direction with the fewest occupied grid points is selected as the routing direction. The algorithm then advances as far as possible in this direction, while marking all intermediate nodes as being occupied. The current node then becomes the new starting node, and the procedure repeats until the total length of the resonator is routed. Because the algorithm advances as far as possible in a given direction, the desired length is often overshot. Therefore, once the length requirement is satisfied, the excess nodes are removed from the resonator path and designated as being unoccupied. 

If the neighboring nodes in every coordinate direction are occupied, the algorithm steps backwards one node at a time and reexamines the surrounding directions. This process continues until an open direction is located. If no open direction exists, then the length requirement cannot be met, and the routing of that particular resonator is aborted. This routing procedure is then repeated for all remaining resonators. Any resonators that are not successfully routed on the first attempt are rerouted after the algorithm has been applied at least once to all cavities. 

The performance of the algorithm is quantified using the routing success rate which is defined as the ratio of successfully routed resonators to the total number of resonators comprising the treatment. 

The computational requirements of this algorithm scale proportionally with the number of resonators. So it may be less suitable for designing treatments for large geometries. Additionally, because the path of each resonator depends on the paths of all previously routed resonators, the algorithm is not easily parallelizable. One potential approach to parallelize it is to subdivide the geometry and route resonators within each section independently; however, this constrains each resonator to its respective subsection.

