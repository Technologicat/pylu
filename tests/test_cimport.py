"""Test that pylu's Cython declarations are importable."""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest


def test_pxd_installed():
    """Verify dgesv.pxd is present in the installed package."""
    import pylu
    pkg_dir = Path(pylu.__file__).parent
    pxd = pkg_dir / "dgesv.pxd"
    assert pxd.exists(), f"dgesv.pxd not found in {pkg_dir}"


def test_cimport_compiles():
    """Verify that 'cimport pylu.dgesv' succeeds in Cython compilation."""
    pytest.importorskip("Cython")
    with tempfile.TemporaryDirectory() as tmpdir:
        pyx = Path(tmpdir) / "test_cimport.pyx"
        pyx.write_text(textwrap.dedent("""\
            cimport pylu.dgesv as dgesv_c
            # If this compiles, the .pxd is discoverable.
        """))
        result = subprocess.run(
            [sys.executable, "-m", "cython", str(pyx)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"cimport pylu.dgesv failed:\n{result.stderr}"
        )
