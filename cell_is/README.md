## Installation of some specific packages

### easydict
The conda command is used here. Find an installation library suitable for your system in [Anaconda](https://anaconda.org/search?q=easydict). The following is the conda command for win64:
```language
conda install -c zhaofeng-shu33 easydict
```

### pycocotools
Download the coco source code from GitHub to any folder, the link is [here](https://github.com/cocodataset/cocoapi).

In the 'cmd' window, 'cd' to the 'PythonAPI' directory, execute the following command:
```python
python setup.py build_ext --inplace                   #install pycocotools locally
python setup.py build_ext install                 # install pycocotools to the Python site-packages
```
Copy the compiled PythonAPI folder to the specified directory.

#### Other errors you may encounter
In Window-System we may meet the error about '/Wno-cpp' or(and) '/Wno-unused-function', open setup-py and delete them.
```python
ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=['-std=c99'],
    )
]
```