"""
extract factors the build is dependent on:
[X] compute capability
    [ ] TODO: Q - What if we have multiple GPUs of different makes?
- CUDA version
- Software:
    - CPU-only: only CPU quantization functions (no optimizer, no matrix multipl)
    - CuBLAS-LT: full-build 8-bit optimizer
    - no CuBLAS-LT: no 8-bit matrix multiplication (`nomatmul`)

evaluation:
    - if paths faulty, return meaningful error
    - else:
        - determine CUDA version
        - determine capabilities
        - based on that set the default path
"""

import ctypes as ct
import os
import errno
import torch
import platform
import site
from warnings import warn
from itertools import product

from pathlib import Path
from typing import Set, Union
from .env_vars import get_potentially_lib_path_containing_env_vars

IS_WINDOWS_PLATFORM: bool = (platform.system()=="Windows")
PATH_COLLECTION_SEPARATOR: str = ":" if not IS_WINDOWS_PLATFORM else ";"

# these are the most common libs names
# libcudart.so is missing by default for a conda install with PyTorch 2.0 and instead
# we have libcudart.so.11.0 which causes a lot of errors before
# not sure if libcudart.so.12.0 exists in pytorch installs, but it does not hurt
CUDA_RUNTIME_LIBS: list = ["libcudart.so", 'libcudart.so.11.0', 'libcudart.so.12.0'] if not IS_WINDOWS_PLATFORM else ["cudart64_110.dll", "cudart64_120.dll", "cudart64_12.dll"]

# this is a order list of backup paths to search CUDA in, if it cannot be found in the main environmental paths
backup_paths = [os.path.join(os.environ.get("CONDA_PREFIX", os.getcwd()), "lib" if not IS_WINDOWS_PLATFORM else "bin", lib) for lib in CUDA_RUNTIME_LIBS]

CUDA_SHARED_LIB_NAME: str = "libcuda.so" if not IS_WINDOWS_PLATFORM else f"{os.environ['SystemRoot']}\\System32\\nvcuda.dll"
SHARED_LIB_EXTENSION: str = ".so" if not IS_WINDOWS_PLATFORM else ".dll"
class CUDASetup:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def generate_instructions(self):
        if getattr(self, 'error', False): return
        print(self.error)
        self.error = True
        if not self.cuda_available:
            self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA library was not detected or CUDA not installed.')
            self.add_log_entry('CUDA SETUP: Solution 1): Your paths are probably not up-to-date. You can update them via: sudo ldconfig.')
            self.add_log_entry('CUDA SETUP: Solution 2): If you do not have sudo rights, you can do the following:')
            self.add_log_entry('CUDA SETUP: Solution 2a): Find the cuda library via: find / -name libcuda.so 2>/dev/null')
            self.add_log_entry('CUDA SETUP: Solution 2b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_2a')
            self.add_log_entry('CUDA SETUP: Solution 2c): For a permanent solution add the export from 2b into your .bashrc file, located at ~/.bashrc')
            self.add_log_entry('CUDA SETUP: Solution 3): For a missing CUDA runtime library (libcudart.so), use `find / -name libcudart.so* and follow with step (2b)')
            return

        if self.cudart_path is None:
            self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA runtime library was not detected.')
            self.add_log_entry('CUDA SETUP: Solution 1: To solve the issue the libcudart.so location needs to be added to the LD_LIBRARY_PATH variable')
            self.add_log_entry('CUDA SETUP: Solution 1a): Find the cuda runtime library via: find / -name libcudart.so 2>/dev/null')
            self.add_log_entry('CUDA SETUP: Solution 1b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_1a')
            self.add_log_entry('CUDA SETUP: Solution 1c): For a permanent solution add the export from 1b into your .bashrc file, located at ~/.bashrc')
            self.add_log_entry('CUDA SETUP: Solution 2: If no library was found in step 1a) you need to install CUDA.')
            self.add_log_entry('CUDA SETUP: Solution 2a): Download CUDA install script: wget https://github.com/TimDettmers/bitsandbytes/blob/main/cuda_install.sh')
            self.add_log_entry('CUDA SETUP: Solution 2b): Install desired CUDA version to desired location. The syntax is bash cuda_install.sh CUDA_VERSION PATH_TO_INSTALL_INTO.')
            self.add_log_entry('CUDA SETUP: Solution 2b): For example, "bash cuda_install.sh 113 ~/local/" will download CUDA 11.3 and install into the folder ~/local')
            return

        make_cmd = f'CUDA_VERSION={self.cuda_version_string}'
        if len(self.cuda_version_string) < 3:
            make_cmd += ' make cuda92'
        elif self.cuda_version_string == '110':
            make_cmd += ' make cuda110'
        elif self.cuda_version_string[:2] == '11' and int(self.cuda_version_string[2]) > 0:
            make_cmd += ' make cuda11x'
        elif self.cuda_version_string == '100':
            self.add_log_entry('CUDA SETUP: CUDA 10.0 not supported. Please use a different CUDA version.')
            self.add_log_entry('CUDA SETUP: Before you try again running bitsandbytes, make sure old CUDA 10.0 versions are uninstalled and removed from $LD_LIBRARY_PATH variables.')
            return


        has_cublaslt = is_cublasLt_compatible(self.cc)
        if not has_cublaslt:
            make_cmd += '_nomatmul'

        self.add_log_entry('CUDA SETUP: Something unexpected happened. Please compile from source:')
        self.add_log_entry('git clone https://github.com/TimDettmers/bitsandbytes.git')
        self.add_log_entry('cd bitsandbytes')
        self.add_log_entry(make_cmd)
        self.add_log_entry('python setup.py install')

    def initialize(self):
        if not getattr(self, 'initialized', False):
            self.has_printed = False
            self.lib = None
            self.initialized = False
            self.error = False

    def manual_override(self):
        if torch.cuda.is_available():
            if 'BNB_CUDA_VERSION' in os.environ:
                if len(os.environ['BNB_CUDA_VERSION']) > 0:
                    warn((f'\n\n{"="*80}\n'
                          'WARNING: Manual override via BNB_CUDA_VERSION env variable detected!\n'
                          'BNB_CUDA_VERSION=XXX can be used to load a bitsandbytes version that is different from the PyTorch CUDA version.\n'
                          f'If this was unintended set the BNB_CUDA_VERSION variable to an empty string: {"set BNB_CUDA_VERSION=" if IS_WINDOWS_PLATFORM else "export BNB_CUDA_VERSION="}\n'
                          'If you use the manual override make sure the right libcudart.so is in your LD_LIBRARY_PATH\n' if not IS_WINDOWS_PLATFORM else ''
                          'For example by adding the following to your .bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_cuda_dir/lib64\n' if not IS_WINDOWS_PLATFORM else ''
                          f'Loading CUDA version: BNB_CUDA_VERSION={os.environ["BNB_CUDA_VERSION"]}'
                          f'\n{"="*80}\n\n'))
                    self.binary_name = self.binary_name[:-6] + f'{os.environ["BNB_CUDA_VERSION"]}' + SHARED_LIB_EXTENSION

    def run_cuda_setup(self):
        self.initialized = True
        self.cuda_setup_log = []

        binary_name, cudart_path, cc, cuda_version_string = evaluate_cuda_setup()
        self.cudart_path = cudart_path
        self.cuda_available = torch.cuda.is_available()
        self.cc = cc
        self.cuda_version_string = cuda_version_string
        self.binary_name = binary_name
        self.manual_override()

        package_dir = Path(__file__).parent.parent
        binary_path = package_dir / self.binary_name

        print('bin', binary_path)

        try:
            if not binary_path.exists():
                self.add_log_entry(f"CUDA SETUP: Required library version not found: {binary_name}. Maybe you need to compile it from source?")
                legacy_binary_name = "libbitsandbytes_cpu" + SHARED_LIB_EXTENSION
                self.add_log_entry(f"CUDA SETUP: Defaulting to {legacy_binary_name}...")
                binary_path = package_dir / legacy_binary_name
                if not binary_path.exists() or torch.cuda.is_available():
                    self.add_log_entry('')
                    self.add_log_entry('='*48 + 'ERROR' + '='*37)
                    self.add_log_entry('CUDA SETUP: CUDA detection failed! Possible reasons:')
                    self.add_log_entry('1. You need to manually override the PyTorch CUDA version. Please see: '
                             '"https://github.com/TimDettmers/bitsandbytes/blob/main/how_to_use_nonpytorch_cuda.md')
                    self.add_log_entry('2. CUDA driver not installed')
                    self.add_log_entry('3. CUDA not installed')
                    self.add_log_entry('4. You have multiple conflicting CUDA libraries')
                    self.add_log_entry('5. Required library not pre-compiled for this bitsandbytes release!')
                    self.add_log_entry('CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=113`.')
                    self.add_log_entry('CUDA SETUP: The CUDA version for the compile might depend on your conda install. Inspect CUDA version via `conda list | grep cuda`.')
                    self.add_log_entry('='*80)
                    self.add_log_entry('')
                    self.generate_instructions()
                    raise Exception('CUDA SETUP: Setup Failed!')
                self.lib = ct.cdll.LoadLibrary(str(binary_path))
            else:
                self.add_log_entry(f"CUDA SETUP: Loading binary {binary_path}...")
                self.lib = ct.cdll.LoadLibrary(str(binary_path))
        except Exception as ex:
            self.add_log_entry(str(ex))

    def add_log_entry(self, msg, is_warning=False):
        self.cuda_setup_log.append((msg, is_warning))

    def print_log_stack(self):
        for msg, is_warning in self.cuda_setup_log:
            if is_warning:
                warn(msg)
            else:
                print(msg)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


def is_cublasLt_compatible(cc):
    has_cublaslt = False
    if cc is not None:
        cc_major, cc_minor = cc.split('.')
        if int(cc_major) < 7 or (int(cc_major) == 7 and int(cc_minor) < 5):
            CUDASetup.get_instance().add_log_entry("WARNING: Compute capability < 7.5 detected! Only slow 8-bit matmul is supported for your GPU!", is_warning=True)
        else:
            has_cublaslt = True
    return has_cublaslt

def extract_candidate_paths(paths_list_candidate: str) -> Set[Path]:
    return {Path(ld_path) for ld_path in paths_list_candidate.split(PATH_COLLECTION_SEPARATOR) if ld_path}


def remove_non_existent_dirs(candidate_paths: Set[Path]) -> Set[Path]:
    existent_directories: Set[Path] = set()
    for path in candidate_paths:
        try:
            if os.path.isdir(path):
                existent_directories.add(path)
        except OSError as exc:
            if exc.errno != errno.ENAMETOOLONG:
                raise exc
        except PermissionError as pex:
            pass

    non_existent_directories: Set[Path] = candidate_paths - existent_directories
    if non_existent_directories:
        CUDASetup.get_instance().add_log_entry("The following directories listed in your path were found to "
            f"be non-existent: {non_existent_directories}", is_warning=False)

    return existent_directories


def get_cuda_runtime_lib_paths(candidate_paths: Set[Path]) -> Set[Path]:
    paths = set()
    for libname in CUDA_RUNTIME_LIBS:
        for path in candidate_paths:
            if (path / libname).is_file():
                paths.add(path / libname)
    return paths


def resolve_paths_list(paths_list_candidate: str) -> Set[Path]:
    """
    Searches a given environmental var for the CUDA runtime library,
    i.e. `libcudart.so`.
    """
    return remove_non_existent_dirs(extract_candidate_paths(paths_list_candidate))


def find_cuda_lib_in(paths_list_candidate: str) -> Set[Path]:
    return get_cuda_runtime_lib_paths(
        resolve_paths_list(paths_list_candidate)
    )


def warn_in_case_of_duplicates(results_paths: Set[Path]) -> None:
    if len(results_paths) > 1:
        warning_msg = (
            f"Found duplicate {CUDA_RUNTIME_LIBS} files: {results_paths}.. "
            f"We select the PyTorch default {'libcudart.so' if not IS_WINDOWS_PLATFORM else 'cudart64_*.dll'}, which is {torch.version.cuda},"
            "but this might missmatch with the CUDA version that is needed for bitsandbytes."
            "To override this behavior set the BNB_CUDA_VERSION=<version string, e.g. 122> environmental variable"
            "For example, if you want to use the CUDA version 122:"
            "BNB_CUDA_VERSION=122 python ..." if not IS_WINDOWS_PLATFORM else "set BNB_CUDA_VERSION=122\npython ..."
            "OR set the environmental variable in your .bashrc: export BNB_CUDA_VERSION=122" if not IS_WINDOWS_PLATFORM else ''
            "In the case of a manual override, make sure you set the LD_LIBRARY_PATH, e.g." if not IS_WINDOWS_PLATFORM else ''
            "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2" if not IS_WINDOWS_PLATFORM else '')
        CUDASetup.get_instance().add_log_entry(warning_msg, is_warning=True)


def determine_cuda_runtime_lib_path() -> Union[Path, None]:
    """
        Searches for a cuda installations, in the following order of priority:
            1. active conda env
            2. LD_LIBRARY_PATH
            3. any other env vars, while ignoring those that
                - are known to be unrelated (see `bnb.cuda_setup.env_vars.to_be_ignored`)
                - don't contain the path separator `/`

        If multiple libraries are found in part 3, we optimistically try one,
        while giving a warning message.
    """
    candidate_env_vars = get_potentially_lib_path_containing_env_vars()

    if "CONDA_PREFIX" in candidate_env_vars:
        conda_libs_path = Path(candidate_env_vars["CONDA_PREFIX"]) / "bin"

        conda_cuda_libs = find_cuda_lib_in(str(conda_libs_path))

        if conda_cuda_libs:
            warn_in_case_of_duplicates(conda_cuda_libs)
            return next(iter(conda_cuda_libs))
        
        conda_libs_path = Path(candidate_env_vars["CONDA_PREFIX"]) / "lib"

        conda_cuda_libs = find_cuda_lib_in(str(conda_libs_path))

        if conda_cuda_libs:
            warn_in_case_of_duplicates(conda_cuda_libs)
            return next(iter(conda_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CONDA_PREFIX"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)

    for sitedir in site.getsitepackages():
        if "site-packages" in sitedir:
                site_packages_path = sitedir
                break
    if site_packages_path:
        torch_libs_path = os.path.join(site_packages_path, "torch", "lib")
        
        if os.path.isdir(torch_libs_path):
            torch_cuda_libs = find_cuda_lib_in(str(torch_libs_path))

            if torch_cuda_libs:
                warn_in_case_of_duplicates(torch_cuda_libs)
                return next(iter(torch_cuda_libs))

            CUDASetup.get_instance().add_log_entry(f'{torch_cuda_libs} did not contain '
                f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
        
    if "CUDA_PATH" in candidate_env_vars:
        win_toolkit_libs_path = Path(candidate_env_vars["CUDA_PATH"]) / "bin"
    
        win_toolkit_cuda_libs = find_cuda_lib_in(str(win_toolkit_libs_path))

        if win_toolkit_cuda_libs:
            warn_in_case_of_duplicates(win_toolkit_cuda_libs)
            return next(iter(win_toolkit_cuda_libs))

        win_toolkit_libs_path = Path(candidate_env_vars["CUDA_PATH"]) / "lib"
    
        win_toolkit_cuda_libs = find_cuda_lib_in(str(win_toolkit_libs_path))

        if win_toolkit_cuda_libs:
            warn_in_case_of_duplicates(win_toolkit_cuda_libs)
            return next(iter(win_toolkit_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CUDA_PATH"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
        
    if "CUDA_HOME" in candidate_env_vars:
        lin_toolkit_libs_path = Path(candidate_env_vars["CUDA_HOME"]) / "bin"
    
        lin_toolkit_cuda_libs = find_cuda_lib_in(str(lin_toolkit_libs_path))

        if lin_toolkit_cuda_libs:
            warn_in_case_of_duplicates(lin_toolkit_cuda_libs)
            return next(iter(lin_toolkit_cuda_libs))
        
        lin_toolkit_libs_path = Path(candidate_env_vars["CUDA_HOME"]) / "lib"
    
        lin_toolkit_cuda_libs = find_cuda_lib_in(str(lin_toolkit_libs_path))

        if lin_toolkit_cuda_libs:
            warn_in_case_of_duplicates(lin_toolkit_cuda_libs)
            return next(iter(lin_toolkit_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CUDA_HOME"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)

    if "LD_LIBRARY_PATH" in candidate_env_vars:
        lib_ld_cuda_libs = find_cuda_lib_in(candidate_env_vars["LD_LIBRARY_PATH"])

        if lib_ld_cuda_libs:
            warn_in_case_of_duplicates(lib_ld_cuda_libs)
            return next(iter(lib_ld_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["LD_LIBRARY_PATH"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
        
    if "PATH" in candidate_env_vars:
        lib_path_cuda_libs = find_cuda_lib_in(candidate_env_vars["PATH"])

        if lib_path_cuda_libs:
            warn_in_case_of_duplicates(lib_path_cuda_libs)
            return next(iter(lib_path_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["PATH"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
        
    remaining_candidate_env_vars = {
        env_var: value for env_var, value in candidate_env_vars.items()
        if env_var not in {"CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PATH"}
    }

    possible_cuda_runtime_libs = set()
    for env_var, value in remaining_candidate_env_vars.items():
        possible_cuda_runtime_libs.update(find_cuda_lib_in(value))

    if len(possible_cuda_runtime_libs) == 0:
        CUDASetup.get_instance().add_log_entry(f'CUDA_SETUP: WARNING! {CUDA_RUNTIME_LIBS} not found in any environmental path. Searching in backup paths...')
        backup_cuda_libs = [find_cuda_lib_in(os.path.realpath(backup_path)) for backup_path in backup_paths if os.path.isdir(backup_path)]
        if backup_cuda_libs:
            possible_cuda_runtime_libs.update(backup_cuda_libs)

    warn_in_case_of_duplicates(possible_cuda_runtime_libs)

    cuda_setup = CUDASetup.get_instance()
    cuda_setup.add_log_entry(f'DEBUG: Possible options found for libcudart.so: {possible_cuda_runtime_libs}')

    return next(iter(possible_cuda_runtime_libs)) if possible_cuda_runtime_libs else None


# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION
def get_cuda_version():
    major, minor = map(int, torch.version.cuda.split("."))

    if major < 11:
        CUDASetup.get_instance().add_log_entry('CUDA SETUP: CUDA version lower than 11 are currently not supported for LLM.int8(). You will be only to use 8-bit optimizers and quantization routines!!')

    return f'{major}{minor}'

def get_compute_capabilities():
    ccs = []
    for i in range(torch.cuda.device_count()):
        cc_major, cc_minor = torch.cuda.get_device_capability(torch.cuda.device(i))
        ccs.append(f"{cc_major}.{cc_minor}")

    return ccs


def evaluate_cuda_setup():
    cuda_setup = CUDASetup.get_instance()
    if 'BITSANDBYTES_NOWELCOME' not in os.environ or str(os.environ['BITSANDBYTES_NOWELCOME']) == '0':
        cuda_setup.add_log_entry('')
        cuda_setup.add_log_entry('='*35 + 'BUG REPORT' + '='*35)
        cuda_setup.add_log_entry(('Welcome to bitsandbytes. For bug reports, please run\n\npython -m bitsandbytes\n\n'),
              ('and submit this information together with your error trace to: https://github.com/jllllll/bitsandbytes/issues'))
        cuda_setup.add_log_entry('='*80)
        return 'libbitsandbytes_cuda118.dll', None, None, None
    if not torch.cuda.is_available(): return 'libbitsandbytes_cpu.so', None, None, None

    cudart_path = determine_cuda_runtime_lib_path()
    ccs = get_compute_capabilities()
    ccs.sort()
    cc = ccs[-1] # we take the highest capability
    cuda_version_string = get_cuda_version()

    cuda_setup.add_log_entry(f"CUDA SETUP: PyTorch settings found: CUDA_VERSION={cuda_version_string}, Highest Compute Capability: {cc}.")
    cuda_setup.add_log_entry(f"CUDA SETUP: To manually override the PyTorch CUDA version please see:"
                             "https://github.com/TimDettmers/bitsandbytes/blob/main/how_to_use_nonpytorch_cuda.md")


    # 7.5 is the minimum CC vor cublaslt
    has_cublaslt = is_cublasLt_compatible(cc)

    # TODO:
    # (1) CUDA missing cases (no CUDA installed by CUDA driver (nvidia-smi accessible)
    # (2) Multiple CUDA versions installed

    # we use ls -l instead of nvcc to determine the cuda version
    # since most installations will have the libcudart.so installed, but not the compiler

    if has_cublaslt:
        binary_name = f"libbitsandbytes_cuda{cuda_version_string}" + SHARED_LIB_EXTENSION
    else:
        "if not has_cublaslt (CC < 7.5), then we have to choose  _nocublaslt"
        binary_name = f"libbitsandbytes_cuda{cuda_version_string}_nocublaslt" +  SHARED_LIB_EXTENSION

    return binary_name, cudart_path, cc, cuda_version_string
