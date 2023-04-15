Install
=====

.. _installation:

Before all
------------


These project does not contain requirements for PyTorch. Because the versions of them depend on your environment. Please install PyTorch at first (see installation guide below.) 

The scripts are tested with PyTorch 1.12.1 and 1.13.0, Diffusers 0.10.2.

For pytorch and xformers,other versions of PyTorch and xformers seem to have problems with training.
If there is no other reason, please install the specified version.

Windows installation
------------

Required Dependencies
^^^^^^^^^^^^^^^^^^^

To install kohya-ss on Windows, you need to install Python 3.10.6 or later:

- Python 3.10.6: `https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe`
- git: `https://git-scm.com/download/win`
  
Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type ``Set-ExecutionPolicy Unrestricted`` and answer A
- Close admin powershell window

Setup the project
^^^^^^^^^^^^^^^^^^^

Open a regular Powershell terminal and type the following inside:

.. code-block:: powershell

   git clone https://github.com/kohya-ss/sd-scripts.git
   cd sd-scripts

   python -m venv venv
   .\venv\Scripts\activate

   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
   pip install --upgrade -r requirements.txt
   pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

   cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
   cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
   cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

   accelerate config

update: ``python -m venv venv`` is seemed to be safer than ``python -m venv --system-site-packages venv`` (some user have packages in global python).

Answers to accelerate config:

.. code-block:: txt

  - This machine
  - No distributed training
  - NO
  - NO
  - NO
  - all
  - fp16

Note: Some users report that a "ValueError: fp16 mixed precision requires a GPU" occurs during training. In this case, answer `0` for the 6th question:
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:``

(Single GPU with id `0` will be used.)

Upgrade the project
^^^^^^^^^^^^^^^^^^^


When a new release comes out you can upgrade your repo with the following command:

.. code-block:: powershell

   cd sd-scripts
   git pull
   .\venv\Scripts\activate
   pip install --use-pep517 --upgrade -r requirements.txt


Once the commands have completed successfully you should be ready to use the new version.


Linux Installation
------------------

TODO