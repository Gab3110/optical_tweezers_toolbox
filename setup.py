#!/usr/bin/env python3
"""
Setup helper for optical_tweezers_toolbox.

What it does:
  1. Scans the toolbox scripts for third-party Python imports.
  2. Installs required conda build prerequisites when needed.
  3. Installs the required Python packages into the current Python environment.
  4. Ensures sfHMM contains the custom CB_GaussStep class and exports it from sfHMM.step.
  5. Rewrites toolbox sys.path.insert(...) entries so they point to this folder.
  6. Installs Spyder as a final conda dependency if it is missing.

Typical use, from inside the toolbox folder:
  python setup.py

Useful alternatives:
  python setup.py --dry-run
  python setup.py --paths-only
  python setup.py --deps-only
  python setup.py --upgrade
  python setup.py --skip-spyder

Important:
  Run this with the Python/conda environment you actually want to use for analysis.
  For prox-tv, this script installs LAPACK/OpenBLAS/compiler prerequisites into
  the current conda environment before running pip install prox-tv.
  By default, it also checks for Spyder at the end and installs spyder with conda
  if it is not already importable. Use --skip-spyder to skip that final dependency.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dependency:
    """Map an imported module name to the pip package/spec that installs it."""

    import_key: str          # top-level import seen in the scripts, e.g. sklearn
    import_check: str        # module that must be importable after install, e.g. sklearn
    pip_spec: str            # pip install argument, e.g. scikit-learn or git+https://...
    label: str | None = None # nicer name for printing/requirements comments

    @property
    def display_name(self) -> str:
        return self.label or self.pip_spec


SCRIPT_GLOBS: tuple[str, ...] = (
    "PM/*.py",
    "fecs/*.py",
    "tweezers_toolbox_modules/*.py",
)

EXCLUDE_FILES = {
    "setup.py",
    "setup_polished.py",
    "inverse_setup.py",
}

LOCAL_IMPORTS = {
    "tweezers_toolbox_modules",
}

# Dependency mapping from the imports currently used by the toolbox.
# sfHMM is intentionally installed from GitHub, not PyPI.
DEPENDENCY_MAP: dict[str, Dependency] = {
    "numpy": Dependency("numpy", "numpy", "numpy"),
    "pandas": Dependency("pandas", "pandas", "pandas"),
    "matplotlib": Dependency("matplotlib", "matplotlib", "matplotlib"),
    "scipy": Dependency("scipy", "scipy", "scipy"),
    "sklearn": Dependency("sklearn", "sklearn", "scikit-learn"),
    "statsmodels": Dependency("statsmodels", "statsmodels", "statsmodels"),
    "seaborn": Dependency("seaborn", "seaborn", "seaborn"),
    "IPython": Dependency("IPython", "IPython", "ipython"),
    "dill": Dependency("dill", "dill", "dill"),
    "networkx": Dependency("networkx", "networkx", "networkx"),
    "ruptures": Dependency("ruptures", "ruptures", "ruptures"),
    "kneed": Dependency("kneed", "kneed", "kneed"),
    "hampel": Dependency("hampel", "hampel", "hampel"),
    "lumicks": Dependency("lumicks", "lumicks.pylake", "lumicks.pylake"),
    "prox_tv": Dependency("prox_tv", "prox_tv", "prox-tv"),
    "sfHMM": Dependency(
        "sfHMM",
        "sfHMM",
        "git+https://github.com/hanjinliu/sfHMM.git",
        label="sfHMM from GitHub",
    ),
}

# Keep installs stable/readable instead of arbitrary dictionary order.
DEPENDENCY_ORDER: tuple[str, ...] = (
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "sklearn",
    "statsmodels",
    "seaborn",
    "IPython",
    "dill",
    "networkx",
    "ruptures",
    "kneed",
    "hampel",
    "lumicks",
    "prox_tv",
    "sfHMM",
)

# prox-tv builds a native extension and needs the LAPACKE header. In conda
# environments, install these before pip install prox-tv. The command is
# targeted at sys.prefix so it modifies the Python environment running setup.py,
# not accidentally base.
PROX_TV_CONDA_PREREQS: tuple[str, ...] = (
    "lapack",
    "openblas",
    "compilers",
)

# Installed at the very end as a normal dependency.
# If Spyder is already importable, setup passes and does nothing.
# Do not list spyder-kernels/ipykernel/etc. here; conda will resolve Spyder's
# own dependencies when it installs the full spyder package.
SPYDER_CONDA_PACKAGES: tuple[str, ...] = (
    "spyder",
)

# Toolbox paths are normalized structurally with Python's AST below.
# We intentionally do not guess what the old path looks like. Any
# sys.path.insert(1, <anything>) in a toolbox script is rewritten to the
# actual folder containing this setup.py.
#
# The replacement path is emitted with repr(str(root)), which is safe on
# macOS/Linux and escapes Windows backslashes correctly.


# Custom sfHMM extension used by this toolbox.  sfHMM is installed from the
# upstream GitHub repository, then this class is injected into sfHMM/step/step.py
# if it is not already present.  Keeping the patch here makes setup.py sufficient
# for both fresh installations and already-existing environments.
CB_GAUSSSTEP_SOURCE = r'''## ADDED class for modified KV stepping
class CB_GaussStep(BaseStep):
    def __init__(self, data, p=-1, PF=1.0):
        """
        Gauss-distribution step finding.

        Reference
        ---------
        - Kalafut, B., & Visscher, K. (2008). An objective, model-independent method for
          detection of non-uniform steps in noisy signals. Computer Physics Communications,
          179(10), 716–723. https://doi.org/10.1016/j.cpc.2008.06.008
        - Chistol, G., Liu, S., Hetherington, C. L., Moffitt, J. R., Grimes, S.,
          Jardine, P. J., & Bustamante, C. (2012).
          High degree of coordination and division of labor
          among subunits in a homomeric ring ATPase. Cell, 151(5), 1017–1028.
          https://doi.org/10.1016/j.cell.2012.10.031

        Parameters
        ----------
        data : array
            Input array.
        p : float, optional
            Probability of transition (signal change). If not in a proper range 0<p<0.5,
            then this algorithm will be identical to the original Kalafut-Visscher's.
        PF : float, optional
            Penalty factor for the SIC term. PF=1.0 reproduces the original
            Kalafut–Visscher criterion. PF>1 increases the penalty per step
            (as in the φ29 packaging paper).
        """
        super().__init__(np.asarray(data, dtype=np.float64), p)

        # Store original penalty (for reference/debugging if you want)
        self.penalty_base = self.penalty

        # Apply penalty factor. PF=1.0 -> no change.
        self.PF = PF
        self.penalty = self.PF * self.penalty_base

    def multi_step_finding(self):
        g = GaussMoment().init(self.data)
        self.fit = np.full(self.len, g.total[0] / self.len)
        chi2 = g.chi2  # initialize total chi^2
        heap = Heap()  # chi^2 change (<0), dx, x0, GaussMoment object of the step
        heap.push(g.get_optimal_splitter() + (0, g))

        while True:
            dchi2, dx, x0, g = heap.pop()
            dlogL = self.penalty - self.len / 2 * np.log(1 + dchi2 / chi2)

            if dlogL > 0:
                x = x0 + dx
                g1, g2 = g.split(dx)
                len(g1) > 2 and heap.push(g1.get_optimal_splitter() + (x0, g1))
                len(g2) > 2 and heap.push(g2.get_optimal_splitter() + (x, g2))
                self.step_list.append(x)
                chi2 += dchi2
            else:
                break

        self._finalize()
        return self
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install toolbox dependencies and rewrite toolbox paths."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change/install, but do not modify files or install packages.",
    )
    parser.add_argument(
        "--deps-only",
        action="store_true",
        help="Only install missing Python dependencies; do not rewrite paths.",
    )
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Only rewrite toolbox paths; do not install dependencies.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Pass --upgrade to pip when installing missing dependencies.",
    )
    parser.add_argument(
        "--user",
        action="store_true",
        help="Pass --user to pip. Do not use this inside most conda environments.",
    )
    parser.add_argument(
        "--skip-conda-prereqs",
        action="store_true",
        help="Skip conda build prerequisites such as LAPACK/OpenBLAS/compilers for prox-tv.",
    )
    parser.add_argument(
        "--skip-spyder",
        action="store_true",
        help="Do not install Spyder at the end of setup.",
    )
    parser.add_argument(
        "--write-requirements",
        action="store_true",
        help="Write requirements.txt from the dependencies detected in the scripts.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def iter_toolbox_scripts(root: Path) -> list[Path]:
    scripts: list[Path] = []
    for pattern in SCRIPT_GLOBS:
        scripts.extend(root.glob(pattern))

    clean_scripts: list[Path] = []
    for script in scripts:
        if script.name in EXCLUDE_FILES:
            continue
        if script.name.startswith("._"):
            continue
        if "__pycache__" in script.parts:
            continue
        clean_scripts.append(script)

    return sorted(set(clean_scripts))


def imported_top_level_modules(script: Path) -> set[str]:
    source = script.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(script))
    except SyntaxError as exc:
        raise SystemExit(f"Could not parse {script}: {exc}") from exc

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


def stdlib_names() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", set()))
    # Fallback/extra names that can appear in older Python versions.
    names.update({
        "colorsys", "concurrent", "copy", "datetime", "difflib", "glob",
        "os", "re", "shutil", "sys",
    })
    return names


def scan_external_imports(root: Path) -> tuple[set[str], set[str]]:
    scripts = iter_toolbox_scripts(root)
    if not scripts:
        raise SystemExit(
            "No toolbox scripts found. Expected files matching: " + ", ".join(SCRIPT_GLOBS)
        )

    seen: set[str] = set()
    for script in scripts:
        seen.update(imported_top_level_modules(script))

    external = seen - stdlib_names() - LOCAL_IMPORTS
    mapped = {name for name in external if name in DEPENDENCY_MAP}
    unknown = external - mapped
    return mapped, unknown


def dependencies_used_by_scripts(root: Path) -> list[Dependency]:
    mapped, unknown = scan_external_imports(root)

    deps = [DEPENDENCY_MAP[key] for key in DEPENDENCY_ORDER if key in mapped]
    extras = sorted(mapped - set(DEPENDENCY_ORDER))
    deps.extend(DEPENDENCY_MAP[key] for key in extras)

    print("Third-party imports detected:")
    for dep in deps:
        print(f"  {dep.import_key:12s} -> {dep.display_name}")

    if unknown:
        print("\nWARNING: imports detected but not mapped to pip packages:")
        for name in sorted(unknown):
            print(f"  {name}")
        print("Add these to DEPENDENCY_MAP if they are not local or standard-library modules.")

    return deps


def _ast_pos_to_index(source: str, lineno: int, col_offset: int) -> int:
    """Convert an AST UTF-8 byte offset to a normal Python string index."""
    lines = source.splitlines(keepends=True)
    line = lines[lineno - 1]

    # AST col_offset/end_col_offset are UTF-8 byte offsets.
    prefix = line.encode("utf-8")[:col_offset].decode("utf-8")
    return sum(len(x) for x in lines[:lineno - 1]) + len(prefix)


def _sys_path_insert_one_calls(source: str, filename: str) -> list[ast.Call]:
    """Return every sys.path.insert(1, ...) call in source."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise SystemExit(
            f"Could not parse {filename} while normalizing toolbox paths: {exc}"
        ) from exc

    calls: list[ast.Call] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        is_sys_path_insert = (
            isinstance(func, ast.Attribute)
            and func.attr == "insert"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "path"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "sys"
        )

        if not is_sys_path_insert:
            continue

        # list.insert requires positional arguments. We only normalize index 1,
        # which is the toolbox convention used by these scripts.
        if len(node.args) < 2:
            continue

        first_arg = node.args[0]
        if not (
            isinstance(first_arg, ast.Constant)
            and isinstance(first_arg.value, int)
            and not isinstance(first_arg.value, bool)
            and first_arg.value == 1
        ):
            continue

        if not hasattr(node, "end_lineno") or node.end_lineno is None:
            raise SystemExit(
                f"Python did not provide source positions for {filename}. "
                "Use Python 3.8 or newer."
            )

        calls.append(node)

    return calls


def _normalize_toolbox_path_source(
    source: str,
    root: Path,
    filename: str,
) -> tuple[str, int]:
    """
    Rewrite every sys.path.insert(1, <anything>) to this setup.py directory.

    This deliberately accepts ANY existing second argument, including:
        '/old/mac/path'
        r'C:\\old\\windows\\path'
        {TOOLS_PATH}
        TOOLS_PATH
        SOME_OLD_VARIABLE
        Path(...)
        os.path.join(...)

    The generated path uses repr(str(root)), so Windows backslashes are escaped
    safely without requiring a raw-string prefix.
    """
    calls = _sys_path_insert_one_calls(source, filename)
    if not calls:
        return source, 0

    replacement = f"sys.path.insert(1, {repr(str(root))})"

    # Replace from the end of the file backwards so earlier AST positions stay valid.
    edits: list[tuple[int, int]] = []
    for node in calls:
        start = _ast_pos_to_index(source, node.lineno, node.col_offset)
        end = _ast_pos_to_index(source, node.end_lineno, node.end_col_offset)
        edits.append((start, end))

    updated = source
    for start, end in sorted(edits, reverse=True):
        updated = updated[:start] + replacement + updated[end:]

    return updated, len(edits)


def _verify_toolbox_paths(source: str, root: Path, filename: str) -> int:
    """
    Verify that every sys.path.insert(1, ...) now contains exactly the actual
    toolbox directory as a string literal. Returns the number of verified calls.
    """
    calls = _sys_path_insert_one_calls(source, filename)
    expected = str(root)

    for node in calls:
        second_arg = node.args[1]
        if not (
            isinstance(second_arg, ast.Constant)
            and isinstance(second_arg.value, str)
            and second_arg.value == expected
        ):
            bad = ast.get_source_segment(source, node) or "sys.path.insert(1, ...)"
            raise SystemExit(
                f"Path normalization FAILED in {filename}.\n"
                f"Expected the toolbox path {expected!r}, but found:\n  {bad}"
            )

    return len(calls)


def rewrite_toolbox_paths(root: Path, dry_run: bool = False) -> None:
    """
    Normalize toolbox sys.path.insert(1, ...) calls to the directory containing
    this setup.py, regardless of what path/value was there before.
    """
    root = root.resolve()
    scripts = iter_toolbox_scripts(root)

    changed: list[tuple[Path, int]] = []
    already_correct: list[Path] = []
    no_path_entry: list[Path] = []

    for script in scripts:
        original = script.read_text(encoding="utf-8", errors="replace")
        updated, n_calls = _normalize_toolbox_path_source(
            original,
            root,
            filename=str(script),
        )

        if n_calls == 0:
            no_path_entry.append(script)
            continue

        if updated != original:
            changed.append((script, n_calls))

            if not dry_run:
                script.write_text(updated, encoding="utf-8")

                # Read the file back from disk and verify the actual saved result.
                saved = script.read_text(encoding="utf-8", errors="replace")
                verified = _verify_toolbox_paths(saved, root, filename=str(script))

                if verified != n_calls:
                    raise SystemExit(
                        f"Path verification count mismatch in {script}: "
                        f"rewrote {n_calls}, verified {verified}."
                    )
        else:
            # Even if no text changed, verify that it is genuinely correct.
            _verify_toolbox_paths(original, root, filename=str(script))
            already_correct.append(script)

    print(f"Toolbox folder used for sys.path: {root}")
    print(f"Found {len(scripts)} toolbox Python scripts.")
    print(f"Files needing path correction: {len(changed)}")

    for script, n in changed:
        print(f"  update: {script.relative_to(root)} ({n} sys.path.insert call(s))")

    if already_correct:
        print(f"Already correct: {len(already_correct)} script(s).")

    if no_path_entry:
        print(f"No sys.path.insert(1, ...) present: {len(no_path_entry)} script(s).")

    if dry_run:
        print("Dry run: no files were modified.")
    else:
        # Final full verification pass over every script that has the call.
        total_verified = 0
        for script in scripts:
            saved = script.read_text(encoding="utf-8", errors="replace")
            total_verified += _verify_toolbox_paths(saved, root, filename=str(script))

        print(
            f"Path verification passed: {total_verified} sys.path.insert(1, ...) "
            f"call(s) point to {root}"
        )


def import_is_available(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def missing_dependencies(root: Path) -> list[Dependency]:
    deps = dependencies_used_by_scripts(root)
    missing = [dep for dep in deps if not import_is_available(dep.import_check)]
    return missing


def _patch_sfhmm_files(step_py: Path, init_py: Path, dry_run: bool = False) -> bool:
    """Ensure the installed sfHMM.step package exports CB_GaussStep.

    Returns True when a change is (or, in dry-run mode, would be) required.
    The patch is idempotent: rerunning setup.py will not duplicate the class or
    either export entry.
    """
    if not step_py.is_file() or not init_py.is_file():
        raise SystemExit(
            "sfHMM is installed, but the expected files were not found:\n"
            f"  {step_py}\n"
            f"  {init_py}\n"
            "The upstream sfHMM package layout may have changed."
        )

    step_source = step_py.read_text(encoding="utf-8", errors="replace")
    init_source = init_py.read_text(encoding="utf-8", errors="replace")

    updated_step = step_source
    updated_init = init_source
    changes: list[str] = []

    # 1) Add the class to sfHMM/step/step.py if it is missing.
    if re.search(r"^class\s+CB_GaussStep\s*\(", step_source, flags=re.MULTILINE) is None:
        marker = "class GaussStep(BaseStep):"
        if marker not in step_source:
            raise SystemExit(
                f"Cannot patch {step_py}: expected marker {marker!r} was not found.\n"
                "The upstream sfHMM implementation may have changed."
            )
        updated_step = step_source.replace(
            marker,
            CB_GAUSSSTEP_SOURCE.rstrip() + "\n\n\n" + marker,
            1,
        )
        changes.append("add CB_GaussStep to sfHMM/step/step.py")

    # 2) Export CB_GaussStep from sfHMM/step/__init__.py.
    # Patch the import tuple and __all__ independently so a partial/manual patch
    # is repaired correctly rather than duplicated.
    import_match = re.search(
        r"from\s+\.step\s+import\s*\((?P<body>.*?)\)",
        updated_init,
        flags=re.DOTALL,
    )
    if import_match is None:
        raise SystemExit(
            f"Cannot patch {init_py}: could not find 'from .step import (...)'.\n"
            "The upstream sfHMM package layout may have changed."
        )
    if re.search(r"\bCB_GaussStep\b", import_match.group("body")) is None:
        body_start = import_match.start("body")
        body = import_match.group("body")
        insertion = "\n    CB_GaussStep," if body.startswith("\n") else "CB_GaussStep, "
        updated_init = updated_init[:body_start] + insertion + updated_init[body_start:]
        changes.append("import CB_GaussStep in sfHMM/step/__init__.py")

    all_match = re.search(
        r"__all__\s*=\s*\[(?P<body>.*?)\]",
        updated_init,
        flags=re.DOTALL,
    )
    if all_match is None:
        raise SystemExit(
            f"Cannot patch {init_py}: could not find '__all__ = [...]'.\n"
            "The upstream sfHMM package layout may have changed."
        )
    if re.search(r"['\"]CB_GaussStep['\"]", all_match.group("body")) is None:
        body_start = all_match.start("body")
        body = all_match.group("body")
        insertion = '\n    "CB_GaussStep",' if body.startswith("\n") else '"CB_GaussStep", '
        updated_init = updated_init[:body_start] + insertion + updated_init[body_start:]
        changes.append("add CB_GaussStep to sfHMM.step.__all__")

    if not changes:
        print("\nsfHMM custom patch already present; no changes needed.")
        return False

    print("\nsfHMM custom patch:")
    for change in changes:
        print(f"  {change}")

    if dry_run:
        print("  Dry run: sfHMM files were not modified.")
        return True

    try:
        step_py.write_text(updated_step, encoding="utf-8")
        init_py.write_text(updated_init, encoding="utf-8")
    except PermissionError as exc:
        raise SystemExit(
            "sfHMM was found, but setup.py does not have permission to modify its installed files.\n"
            f"  {step_py}\n"
            f"  {init_py}\n"
            "Run setup.py from the environment that owns this sfHMM installation."
        ) from exc
    return True


def ensure_sfhmm_cb_gaussstep(dry_run: bool = False) -> None:
    """Patch sfHMM whether it was just installed or was already present."""
    spec = importlib.util.find_spec("sfHMM")
    if spec is None:
        print("\nsfHMM is not installed/importable; custom CB_GaussStep patch skipped.")
        return

    if spec.submodule_search_locations:
        package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    elif spec.origin:
        package_root = Path(spec.origin).resolve().parent
    else:
        raise SystemExit("sfHMM was found, but its installation directory could not be determined.")

    step_dir = package_root / "step"
    step_py = step_dir / "step.py"
    init_py = step_dir / "__init__.py"

    print(f"\nsfHMM package: {package_root}")
    _patch_sfhmm_files(step_py, init_py, dry_run=dry_run)

    if dry_run:
        return

    # Verify the installed package, not merely the text edit.  Reload if sfHMM
    # happened to be imported earlier in this Python process.
    importlib.invalidate_caches()
    try:
        if "sfHMM.step.step" in sys.modules:
            importlib.reload(sys.modules["sfHMM.step.step"])
        if "sfHMM.step" in sys.modules:
            step_module = importlib.reload(sys.modules["sfHMM.step"])
        else:
            step_module = importlib.import_module("sfHMM.step")
    except Exception as exc:
        raise SystemExit(f"sfHMM patch was written, but importing sfHMM.step failed: {exc}") from exc

    if not hasattr(step_module, "CB_GaussStep"):
        raise SystemExit(
            "sfHMM patch was written, but sfHMM.step.CB_GaussStep is still unavailable."
        )

    print("sfHMM custom patch check passed: sfHMM.step.CB_GaussStep is available.")


def ensure_git_available_for_git_dependencies(missing: list[Dependency]) -> None:
    needs_git = any(dep.pip_spec.startswith("git+") for dep in missing)
    if needs_git and shutil.which("git") is None:
        raise SystemExit(
            "A GitHub dependency is missing, but the 'git' command is not available.\n"
            "Install git first, then rerun setup.py."
        )


def conda_executable() -> str | None:
    """Return a conda executable if one is available."""
    # CONDA_EXE is set by conda activation and is the most reliable option.
    import os

    conda = os.environ.get("CONDA_EXE")
    if conda and Path(conda).exists():
        return conda
    return shutil.which("conda")


def install_conda_packages(packages: tuple[str, ...], reason: str, dry_run: bool = False) -> None:
    """Install packages into the conda environment running this setup script."""
    conda = conda_executable()
    cmd_text = "conda install -c conda-forge " + " ".join(packages)

    if conda is None:
        raise SystemExit(
            f"{reason}, but the 'conda' command was not found.\n"
            f"Activate your conda environment and run this first:\n  {cmd_text}\n"
            "Then rerun: python setup.py"
        )

    cmd = [
        conda,
        "install",
        "-y",
        "-p",
        sys.prefix,
        "-c",
        "conda-forge",
        *packages,
    ]

    print(f"\n{reason}:")
    print("  " + " ".join(cmd))
    print("  equivalent manual command:")
    print("  " + cmd_text)

    if dry_run:
        return

    subprocess.check_call(cmd)


def install_conda_prereqs_for_prox_tv(dry_run: bool = False) -> None:
    """Install native build prerequisites required by prox-tv."""
    install_conda_packages(
        PROX_TV_CONDA_PREREQS,
        "Installing conda build prerequisites for prox-tv",
        dry_run=dry_run,
    )


def ensure_spyder_installed(dry_run: bool = False) -> None:
    """Install Spyder as the final conda dependency if it is missing."""
    if import_is_available("spyder"):
        print("\nSpyder is already installed/importable; skipping Spyder install.")
        return

    install_conda_packages(
        SPYDER_CONDA_PACKAGES,
        "Installing Spyder as the final conda dependency",
        dry_run=dry_run,
    )

    if not dry_run and not import_is_available("spyder"):
        raise SystemExit(
            "Spyder was installed by conda, but 'import spyder' still fails. "
            "Check the conda output above."
        )


def pip_install(dep: Dependency, dry_run: bool, upgrade: bool, user: bool) -> None:
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    if user:
        cmd.append("--user")
    cmd.append(dep.pip_spec)

    print(f"\nInstalling {dep.display_name} for import '{dep.import_check}':")
    print("  " + " ".join(cmd))

    if dry_run:
        return

    subprocess.check_call(cmd)


def install_dependencies(root: Path, dry_run: bool = False, upgrade: bool = False, user: bool = False, skip_conda_prereqs: bool = False) -> None:
    missing = missing_dependencies(root)

    if not missing:
        print("\nAll required Python packages are already importable.")
        return

    print("\nMissing imports:")
    for dep in missing:
        print(f"  {dep.import_check:16s} -> pip install {dep.pip_spec}")

    ensure_git_available_for_git_dependencies(missing)

    if dry_run:
        if any(dep.import_key == "prox_tv" for dep in missing) and not skip_conda_prereqs:
            install_conda_prereqs_for_prox_tv(dry_run=True)
        print("\nDry run: dependencies were not installed.")
        return

    for dep in missing:
        if dep.import_key == "prox_tv" and not skip_conda_prereqs:
            install_conda_prereqs_for_prox_tv(dry_run=False)
        pip_install(dep, dry_run=False, upgrade=upgrade, user=user)

    still_missing = missing_dependencies(root)
    if still_missing:
        print("\nWARNING: these imports are still missing after pip finished:")
        for dep in still_missing:
            print(f"  {dep.import_check}  (pip spec: {dep.pip_spec})")
        raise SystemExit(
            "Some dependencies did not become importable. Check the pip output above."
        )

    print("\nDependency check passed: all required imports are now available.")


def write_requirements(root: Path) -> None:
    deps = dependencies_used_by_scripts(root)
    req_path = root / "requirements.txt"
    lines = [
        "# Generated from imports in PM/*.py, fecs/*.py, and tweezers_toolbox_modules/*.py",
        "# sfHMM is installed from GitHub intentionally.",
    ]
    lines.extend(dep.pip_spec for dep in deps)
    req_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {req_path}")


def main() -> None:
    args = parse_args()

    if args.deps_only and args.paths_only:
        raise SystemExit("Use either --deps-only or --paths-only, not both.")

    root = repo_root()
    print(f"Toolbox folder: {root}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}\n")

    if args.write_requirements and not args.dry_run:
        write_requirements(root)
        print()

    if not args.deps_only:
        rewrite_toolbox_paths(root, dry_run=args.dry_run)
        print()

    if not args.paths_only:
        install_dependencies(root, dry_run=args.dry_run, upgrade=args.upgrade, user=args.user, skip_conda_prereqs=args.skip_conda_prereqs)
        ensure_sfhmm_cb_gaussstep(dry_run=args.dry_run)

    if not args.paths_only and not args.skip_spyder:
        ensure_spyder_installed(dry_run=args.dry_run)
    elif args.skip_spyder:
        print("\nSkipping Spyder installation because --skip-spyder was used.")


if __name__ == "__main__":
    main()
