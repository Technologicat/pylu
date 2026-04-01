# libm dependency (dgesv.pxd uses libc.math fabs)
# On Linux this links -lm; on macOS/Windows it's a no-op (math is in libc/MSVCRT).
cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)

# Cython extension module
py.extension_module(
    'dgesv',
    'dgesv.pyx',
    dependencies: [numpy_dep, m_dep],
    install: true,
    subdir: 'pylu',
)

# Install Python files
py.install_sources(
    '__init__.py',
    subdir: 'pylu',
)

# CRITICAL: Install .pxd for downstream cimport
# Without this, pydgq and python-wlsqm cannot cimport pylu.dgesv
py.install_sources(
    'dgesv.pxd',
    subdir: 'pylu',
)
