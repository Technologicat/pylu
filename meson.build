project(
    'pylu',
    'cython', 'c',
    version: files('pylu/VERSION'),
    license: 'BSD-2-Clause',
    meson_version: '>=1.3.0',
    default_options: [
        'buildtype=release',
        'c_std=c11',
        'cython_language=c',
    ],
)

# Ensure <math.h> exposes M_PI and friends on MSVC. No-op on Linux/macOS;
# disarms a Windows-only landmine if any downstream Cython code ever touches
# the math constants.
add_project_arguments('-D_USE_MATH_DEFINES', language: 'c')

# Set Cython language level globally so new .pyx/.pxd files can't silently
# miss it. Replaces per-file `# cython: language_level = 3` directives.
add_project_arguments('-X', 'language_level=3', language: 'cython')

py = import('python').find_installation(pure: false)

# NumPy include directory (needed for Cython compilation against NumPy)
numpy_dep = dependency('numpy')

subdir('pylu')
