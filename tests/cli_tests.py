"""Check the public CLIs and their observable timing/output contract."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

single, grid = map(lambda p: str(Path(p).resolve()), sys.argv[1:])


def run(binary, args, cwd, expected=0):
    result = subprocess.run([binary, *args], cwd=cwd, capture_output=True, text=True)
    assert result.returncode == expected, (args, result.stdout, result.stderr)
    return result


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    cpu = str(min(os.sched_getaffinity(0)))
    common = ['--solver', 'hpipm', '--laps', '0.02', '--repetitions', '3',
              '--no-warmup', '--cpu-affinity', cpu]
    result = run(grid, common, root)
    assert 'Comparable trajectories: 144/144' in result.stdout, result.stdout
    row = next(line.split() for line in result.stdout.splitlines() if line.startswith('hpipm'))
    assert row[1:] == ['100.00', '100.00'], row
    assert list(root.iterdir()) == [], 'Grid utility must write no files'

    for trajectory in ('challenging', 'daocp-worst'):
        output = root / trajectory
        result = run(single, [*common, '--trajectory', trajectory, '--output', str(output), '--no-plots'], root)
        assert json.loads((output / 'metadata.json').read_text())['affinity'] == cpu
        with (output / 'steps.csv').open() as stream:
            samples = list(csv.DictReader(stream))
        groups = {}
        for sample in samples:
            groups.setdefault(int(sample['step']), []).append(float(sample['time_tot_us']))
        assert all(len(values) == 3 for values in groups.values()), 'Three repetitions at every step'
        trace = [sum(values) / len(values) for values in groups.values()]
        summary = json.loads((output / 'summary.json').read_text())['results'][0]
        assert summary['valid'] and summary['completed_runs'] == 3
        assert math.isclose(summary['time_tot_us']['average'], sum(trace) / len(trace), rel_tol=1e-12)
        assert math.isclose(summary['time_tot_us']['max_per_sampling_average'], max(trace), rel_tol=1e-12)
        row = next(line.split() for line in result.stdout.splitlines() if line.startswith('hpipm'))
        assert abs(float(row[2]) - sum(trace) / len(trace)) <= 0.0051
        assert abs(float(row[3]) - max(trace)) <= 0.0051

    for args in (['--trajectory', 'easy'], ['--trajectory', 'all'], ['--paper-grid'],
                 ['--laps', 'nan'], ['--repetitions', '2junk'], ['--solver', 'hpipm,hpipm']):
        run(single, args, root, expected=1)
    run(grid, ['--output', 'forbidden'], root, expected=1)
    run(grid, ['--trajectory', 'challenging'], root, expected=1)

print('CLI, timing aggregation, and grid artifact checks passed.')
