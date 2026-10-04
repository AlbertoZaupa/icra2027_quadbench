# Scope
This repository implements one of the benchmarks described in [the paper](https://arxiv.org/abs/2609.33451) presenting [DAOCP](https://github.com/AlbertoZaupa/daocp). Running the code requires [this Acados fork](https://github.com/AlbertoZaupa/acados).

# Quadrotor QP solver benchmark

Two C++ utilities compare DAOCP, HPIPM, DAQP, and qpOASES inside the same
acados SQP-RTI quadrotor MPC controller: 100 Hz updates, 20 shooting intervals,
and a 1 s prediction horizon. Each solver drives its own deterministic closed-loop
simulation with an ideal plant and perfect state feedback.

The model and trajectory family are inspired by Sun et al.,
[*A Comparative Study of Nonlinear MPC and Differential-Flatness-Based Control
for Quadrotor Agile Flight*](https://arxiv.org/abs/2109.01365).

## Requirements and build

Install [this Acados fork](https://github.com/AlbertoZaupa/acados). Use Linux, CMake 3.18+, a C++17 compiler, and Python with the packages in
`requirements.txt` (CasADi for model generation; NumPy and Matplotlib for plots).
Acados must already be installed with its development headers, including
HPIPM, BLASFEO, and DAOCP headers. To run all four solvers, its build must enable
`ACADOS_WITH_DAOCP`, `ACADOS_WITH_DAQP`, and `ACADOS_WITH_QPOASES`.
The benchmark accepts all four solver names even when a backend is disabled;
selecting a disabled backend reports an error.

Export the installation's shared library directory in `LD_LIBRARY_PATH`. For example, in `~/.bashrc`:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
```

CMake discovers Acados from `ACADOS_SOURCE_DIR`, standard system locations,
or the library directories in `LD_LIBRARY_PATH`. If headers are installed in a
separate prefix, pass `-DACADOS_INSTALL_DIR=/path/to/install` or
`-DCMAKE_PREFIX_PATH=/path/to/install`. No Acados sources are built or modified.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBENCH_PYTHON_EXECUTABLE="$VIRTUAL_ENV/bin/python"
cmake --build build -j
```

The default build produces only `quadrotor_benchmark` and
`quadrotor_grid_benchmark`. Internal tests can be enabled with
`-DBUILD_TESTING=ON` and run with `ctest --test-dir build --output-on-failure`.

## Reference trajectories

The paper's horizontal and vertical ellipses use:

```text
r_max = V_max² / a_max;  r_min = r_max / n;  k = a_max / V_max
horizontal: p(t) = [r_max sin(kt), r_min cos(kt), 5]
vertical:   p(t) = [r_max sin(kt), 0, 5 + r_min cos(kt)]
period = 2π / k
```

Horizontal heading points toward the center; vertical heading is constant.
Analytic derivatives and differential flatness with drag supply attitude, body
rates, and feedforward thrust. Each simulation starts on the reference, with
initial body rates clamped to the controller limits.

The full family contains 144 combinations: two planes, speeds
`{5, 10, 15, 20}` m/s, accelerations `{10, 20, 30, 40, 50, 60}` m/s², and
ellipticities `{1, 2, 5}`. It includes dynamically infeasible references.
Two horizontal cases have benchmark aliases:

| Trajectory | V_max [m/s] | a_max [m/s²] | n |
|---|---:|---:|---:|
| `challenging` | 15 | 30 | 2 |
| `daocp-worst` | 15 | 60 | 1 |

`challenging` is a dyanmically feasible, but relatively challenging trajectory, while `daocp-worst` is the trajectory where DAOCP performs worst relatively to the other solvers. 

## Single trajectory benchmark

```bash
./build/quadrotor_benchmark --solver all --trajectory challenging \
  --laps 5 --repetitions 20 --cpu-affinity 3 --output results/challenging
```

Select one trajectory (`challenging` or `daocp-worst`) and any solver subset,
for example `--solver daocp,hpipm`. The utility prints average and worst case
solve times in microseconds for every selected solver. It writes raw samples,
CSV/JSON summaries, run metadata, consistency checks, and plots to the output
directory. Use `--no-plots` to skip plots.

## Full trajectory benchmark

```bash
./build/quadrotor_grid_benchmark --solver daocp,hpipm \
  --laps 5 --repetitions 20 --cpu-affinity 3
```

Runs the selected solvers on all 144 trajectories. It prints a table with each
solver's percentage of trajectories won by average solve time and by worst case
solve time. Every trajectory has equal weight; exact ties share one win.
Trajectories where any selected solver fails to complete every repetition are
excluded from both percentages, and the comparison count is printed. A failed
comparison gives exit status 2. No result files, directories, or plots are created.

Both utilities default to all solvers, 20 measured repetitions, five laps, and
one unmeasured warm-up per solver/trajectory pair. Each repetition creates a
fresh controller and simulation. Fractional laps are supported; `--no-warmup`
skips warm-up. CPU affinity accepts a CPU number, list, or range, such as
`3`, `2,3`, or `2-3`; choose a CPU allowed by your environment.

Timing uses acados `time_tot`, which covers the MPC solve and excludes reference
updates, diagnostics, plant integration, and file output. First, runtimes at each
simulation step are averaged across repetitions. The reported average is the
mean of that trace; the worst case is its maximum. The first cold solve is
included. This averaged maximum is not a hard real-time bound.
