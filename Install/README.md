# Configure virtual environment
## Using venv
When using venv for setting up virtual environment (**preferred** in terms of Python 3.3 or newer) -->
Reference website: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
### Special case: Using Pycharm as IDE
Please use the **Settings>Project:[project name]>Python Interpreter** to build the virtual environment, which is the 
most convenient way.

FAQs:
> Under the situation that the error in terms of loading the file with the .\env\Scripts\Activate.ps1 rises when 
> switching to the virtual environment --> Reference website: https://blog.csdn.net/qq_41574947/article/details/106939020

## Using virtualenv
When using virtualenv --> 
IF python is not installed on current OS, issue the following commands in terminal
1. `sudo apt-get install python3-pip python3.8` (assume that you want to install python3.8)
2. `ls /usr/bin/python*` (check if python3.8 is successfully installed)

OTHERWISE (by checking it with `python -V`), ignore the commands above and issue the following commands

1. `pip3 install virtualenv`
2. `python -m virtualenv -p python3.8 .env` (if you've installed python3.8; if operated in Linux, python3 may be necessary.)
3. `source .env/bin/activate` (on Linux or Mac) or `.env\Scripts\activate` (on Win)
4. `deactivate` (when quitting)

## Install packages in virtual environment
Reference website: https://pip.pypa.io/en/stable/user_guide/#requirements-files
## Install mmdetection
Official instruction: https://mmdetection.readthedocs.io/en/latest/get_started.html#install-mmdetection.

Useful instruction given by a third party on Win10: https://blog.csdn.net/qq_26755205/article/details/115999553.

The general procedure summarized from the two instructions above is:
> Install CUDA and CUdnn, and then pytorch according to their versions in your virtual environment.

> Check for the right version of wheels of mmcv-full given your cuda version and pytporch version using the following 
> link: https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

> Check the essential python package requirements from 'requirements.txt' of the following repo:
> https://github.com/open-mmlab/mmcv.git. And then copy it to 'requirements.txt' of your own repo and intall it 
> in your defined virtual environment.

> Command `pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html`.
> The compatible version of mmcv-full would be automatically searched and set for you.

> Command `pip install mmdet==<corresponding version to the installed mmcv-full>`.

> Validation the installation using the corresponding [repo](https://github.com/open-mmlab/mmdetection) of 
> corresponding branch version and the code snippet given in the 
> [official instruction](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html).

> Sometimes more adaptations of the [repo](https://github.com/open-mmlab/mmdetection) need to be implemented, which can
> be referenced to the [useful instruction](https://blog.csdn.net/qq_26755205/article/details/115999553) mentioned above.
## Others
### Check CUDA version (cross-platform applicable)
- `nvidia-smi` - This driver API
- `nvcc --version` - This runtime API reports the CUDA runtime version, which actually reflects your expectation.

The difference between them see 
[this link](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi).

### Shortcut keys for convenience in coding
Helpful [links](https://www.pythonforbeginners.com/comments/shortcut-to-comment-out-multiple-lines-in-python) to 
shortcuts for commenting out in different IDEs.

- In Pycharm, press `ctrl+shift+/` to comment out multiple lines.
- In Jupyter Notebook, press `ctrl+/` to comment out multiple lines.
