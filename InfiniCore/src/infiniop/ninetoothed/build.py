import concurrent.futures
import functools
import inspect
import itertools
import math
import os
import pathlib

import ninetoothed
from ninetoothed.aot import _HEADER_PATH

CURRENT_FILE_PATH = pathlib.Path(__file__)

BUILD_DIRECTORY_PATH = (
    CURRENT_FILE_PATH.parent.parent.parent.parent / "build" / "ninetoothed"
)


def build(
    premake,
    constexpr_param_grid,
    caller,
    op_name,
    output_dir,
    num_warps=None,
    num_stages=None,
):
    headers = []
    all_param_names = []
    combinations = []
    launches = []

    total_combinations = _count_param_value_combinations(constexpr_param_grid)
    max_workers = _resolve_max_workers(total_combinations)
    max_inflight = _resolve_max_inflight(max_workers)

    if max_workers == 1:
        for combination in _generate_param_value_combinations(constexpr_param_grid):
            header, param_names, combination, launch = _make(
                premake,
                combination,
                caller,
                op_name,
                output_dir,
                num_warps,
                num_stages,
            )
            headers.append(header)
            all_param_names.append(param_names)
            combinations.append(combination)
            launches.append(launch)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            in_flight = set()
            combination_iter = _generate_param_value_combinations(constexpr_param_grid)

            while True:
                while len(in_flight) < max_inflight:
                    combination = next(combination_iter, None)
                    if combination is None:
                        break
                    in_flight.add(
                        executor.submit(
                            _make,
                            premake,
                            combination,
                            caller,
                            op_name,
                            output_dir,
                            num_warps,
                            num_stages,
                        )
                    )

                if not in_flight:
                    break

                done, _ = concurrent.futures.wait(
                    in_flight, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    in_flight.remove(future)
                    header, param_names, combination, launch = future.result()
                    headers.append(header)
                    all_param_names.append(param_names)
                    combinations.append(combination)
                    launches.append(launch)

    includes = "\n".join(f'#include "{header}"' for header in headers)

    param_names = list(
        functools.reduce(
            lambda x, y: dict.fromkeys(x) | dict.fromkeys(y),
            sorted(all_param_names, key=len, reverse=True),
            {},
        )
    )
    param_types = [
        "NineToothedStream",
    ] + ["NineToothedTensor" for _ in range(len(param_names) - 1)]

    for param_name in functools.reduce(lambda x, y: x | y, combinations, {}):
        param_names.append(param_name)
        param_types.append("int")

    param_decls = ", ".join(
        f"{type} {param}" for param, type in zip(param_names, param_types)
    )

    source_file_name = f"{op_name}.c"
    header_file_name = f"{op_name}.h"

    func_sig = f"NineToothedResult launch_{op_name}({param_decls})"

    joined_launches = "\n".join(launches)

    op_decl = f'#ifdef __cplusplus\nextern "C" {func_sig};\n#else\n{func_sig};\n#endif'
    op_def = f"""{func_sig} {{
{joined_launches}
    return INFINI_STATUS_NOT_IMPLEMENTED;
}}"""

    source_content = f"""#include "{header_file_name}"

#include "infinicore.h"

{includes}\n\n{op_def}\n"""
    header_content = f"""#include "{_HEADER_PATH}"
\n{op_decl}\n"""

    (BUILD_DIRECTORY_PATH / source_file_name).write_text(source_content)
    (BUILD_DIRECTORY_PATH / header_file_name).write_text(header_content)


def _make(premake, combination, caller, op_name, output_dir, num_warps, num_stages):
    arrangement, application, tensors = premake(**combination)

    for param_name, param_value in combination.items():
        if isinstance(param_value, bool):
            combination[param_name] = int(param_value)
            continue
        if isinstance(param_value, str):
            combination[param_name] = (
                f"INFINI_DTYPE_{combination[param_name].replace('fp', 'F').upper()}"
            )

    combination = {f"{name}_": value for name, value in combination.items()}

    kernel_name = f"{op_name}_{_generate_suffix(combination.values())}"

    ninetoothed.make(
        arrangement,
        application,
        tensors,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    header = output_dir / f"{kernel_name}.h"
    param_names = ("stream",) + tuple(inspect.signature(application).parameters.keys())
    launch = f"""    if ({_generate_condition(combination)})
        return launch_{kernel_name}({", ".join(param_names)});"""

    return header, param_names, combination, launch


def _generate_condition(combination):
    return " && ".join(f"{param} == {value}" for param, value in combination.items())


def _generate_suffix(values):
    return "_".join(f"{value}" for value in values)


def _generate_param_value_combinations(param_grid):
    keys = list(param_grid.keys())
    for value_combination in itertools.product(*param_grid.values()):
        yield dict(zip(keys, value_combination))


def _count_param_value_combinations(param_grid):
    return math.prod(len(values) for values in param_grid.values())


def _parse_positive_int_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        return None
    if value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _resolve_max_workers(total_combinations):
    env_max_workers = _parse_positive_int_env("NINETOOTHED_MAX_WORKERS")
    if env_max_workers is not None:
        return max(1, min(env_max_workers, total_combinations))

    cpu_count = os.cpu_count() or 1
    default_max_workers = max(1, min(cpu_count, 4))
    return max(1, min(default_max_workers, total_combinations))


def _resolve_max_inflight(max_workers):
    env_max_inflight = _parse_positive_int_env("NINETOOTHED_MAX_INFLIGHT")
    if env_max_inflight is not None:
        return max(1, env_max_inflight)
    return max(1, max_workers * 2)
