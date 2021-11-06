# Configure virtual environment

IF python is not installed on current OS, issue the following commands in terminal
1. `sudo apt-get install python3-pip python3.8` (assume that you want to install python3.8)
2. `ls /usr/bin/python*` (check if python3.8 is successfully installed)

OTHERWISE (by checking it with `python -V`), ignore the commands above and issue the following commands

1. `pip3 install virtualenv`
2. `python -m virtualenv -p python3.8 .env` (if you've installed python3.8; if operated in Linux, python3 may be necessary.)
3. `source .env/bin/activate` (on Linux or Mac) or `.env\Scripts\activate.bat` (on Win)
4. `deactivate` (when quitting)

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
