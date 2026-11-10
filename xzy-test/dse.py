import accelforge as af
from pathlib import Path

import os
from pathlib import Path

EXAMPLES_DIR = Path(os.path.abspath("../examples"))

examples_dir = Path("../examples")
# First import AccelForge and initialize a Spec. 
# We'll use a 4096x4096x4096 matrix multiply as our workload.
spec = af.Spec.from_yaml(
    examples_dir / "arches" / "eyeriss.yaml",
    examples_dir / "workloads" / "basic" / "matmuls.yaml",
    jinja_parse_data={"N_EINSUMS": 1, "M": 4096, "KN": 4096},
)
spec.mapper.metrics = af.Metrics.ENERGY

# Explore the Global-Buffer
# Define the size range (size_start ~ size_end) of Global-Buffer for explore
size_start = 32 * 8  # 32 bytes
size_end = 4 * 1024 * 1024 * 8  # 4MB
n_sizes = 32  # Number of sizes to explore
sizes = [
    int(size_start * (size_end / size_start) ** (i / (n_sizes - 1))) for i in range(n_sizes)
]
# size_start = 256 * 1024 * 8  # 256KB
# size_end = 4 * 1024 * 1024 * 8  # 4MB
# n_sizes = int(size_end / size_start)
# sizes = [
#     size_start + (size_end/size_start) * i for i in range(n_sizes)
# ]

# Define a function that will map the workload to the architecture
# Set the Global-Buffer size to the given size.
def get_mapper_result(spec: af.Spec, global_buffer_size: int):
    spec.arch.find("GlobalBuffer").size = global_buffer_size
    return spec.map_workload_to_arch(
        print_progress=False, print_number_of_pmappings=False
    )

# < SLOWER >
# Directly use the function
results = {size: get_mapper_result(spec, size) for size in sizes}

# from accelforge.util import parallel, delayed

# spec.mapper.metrics = af.Metrics.ENERGY
# jobs = {size: delayed(get_mapper_result)(spec, size) for size in sizes}
# results = parallel(jobs, pbar="Generating Mappings")

from typing import Any


import matplotlib.pyplot as plt

x = list(results.keys())
y = [v.per_compute().energy() * 1e12 for v in results.values()]
plt.plot(x, y)
plt.xlabel("Global Buffer Size (bits)")
plt.ylabel("Energy per Compute (pJ)")
plt.xscale("log")
plt.show()

x = list(results.keys())
per_component_results = {}
for r in results.values():
    for component, energy in r.per_compute().energy(per_component=True).items():
        per_component_results.setdefault(component, []).append(energy)

for component, energy in per_component_results.items():
    plt.plot(x, energy, label=component)
plt.legend()
plt.xlabel("Global Buffer Size (bits)")
plt.ylabel("Energy (pJ)")
plt.xscale("log")
plt.show()

def get_area(spec: af.Spec, size: int):
    spec.arch.find("GlobalBuffer").size = size
    return spec.calculate_component_costs()


areas = {k: delayed(get_area)(spec, k) for k in sizes}
areas = parallel(areas, pbar="Calculating Area")

# Plot the total area
x = list(areas.keys())
y = [cur_spec.arch.total_area * 1e6 for cur_spec in list(areas.values())]
plt.plot(x, y)
plt.xlabel("Global Buffer Size (bits)")
plt.ylabel("Area (mm^2)")
plt.xscale("log")
plt.yscale("log")
plt.show()

# Plot per-component area
x = list(areas.keys())
y = {}
for cur_spec in list(areas.values()):
    for component, area in cur_spec.arch.per_component_total_area.items():
        y.setdefault(component, []).append(area * 1e6)

for component, area in y.items():
    plt.plot(x, area, label=component)
plt.legend()
plt.xlabel("Global Buffer Size (bits)")
plt.ylabel("Area (mm^2)")
plt.xscale("log")
plt.yscale("log")
plt.show()