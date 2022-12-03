import os
import platform
import subprocess

from setuptools import Extension, dist, find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

dist.Distribution().fetch_build_eggs(['Cython', 'numpy>=1.11.1'])
import numpy as np
from Cython.Build import cythonize


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content



def get_git_hash():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def make_cuda_ext(name, module, sources):
    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args={
            'cxx' : [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


if __name__ == '__main__':
    setup(
        name='mmdet',
        version='1.0.0',
        description='Open MMLab Detection Toolbox and Benchmark',
        author='OpenMMLab',
        author_email='xx@gmail.com',
        keywords='computer vision, object detection',
        url='https://github.com/open-mmlab/mmdetection',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        package_data={'mmdet.ops': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        license='Apache License 2.0',
        setup_requires=['pytest-runner', 'cython', 'numpy'],
        tests_require=['pytest'],
        # install_requires=get_requirements(),
        ext_modules=[
            make_cuda_ext(
                name='deform_conv_cuda',
                module='lib.models.dcn',
                sources=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu'
                ]),
            # make_cuda_ext(
            #     name='deform_pool_cuda',
            #     module='lib.models.dcn',
            #     sources=[
            #         'src/deform_pool_cuda.cpp',
            #         'src/deform_pool_cuda_kernel.cu'
            #     ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)

    # install pysot testing toolkit
    os.chdir('./lib/eval_toolkit/pysot/utils/')
    os.system('python setup.py build_ext --inplace')
