# Try to use setuptools_scm to get the current version; this is only used
# in development installations from the git repository.
import os.path

try:
    from setuptools_scm import get_version

    version = get_version(root=os.path.join("..", ".."), relative_to=__file__)
except ImportError:
    raise
except Exception as e:
    raise ValueError("setuptools_scm can not determine version.") from e
