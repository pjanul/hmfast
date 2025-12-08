Installation
============


To use ``hmfast``, you'll need to install the package and make sure you have the relevant data, such as the emulator files.
The recommended way to get started is as follows:

1. **Install the repository:**

   To install the latest stable version, simply pip install it.

   .. code-block:: bash

      pip install hmfast

   To install the developer version, clone the repository as follows and pip install it locally.

   .. code-block:: bash

      git clone https://github.com/hmfast/hmfast.git
      pip install /your/path/to/hmfast

  

2. **Import the package and begin:**

   You may now import the package. 
   ``hmfast`` relies on pre-trained cosmological emulators to quickly compute halo model quantities. 
   Because these emulator files are too large to include in the package distribution, 
   they will be automatically downloaded the first time you import ``hmfast`` if they are not already present. 

   By default, they are stored in ``~/hmfast_data``.  

   If you want to use a different location for these files, simply uncomment the first two lines below and set your preferred path.

   .. code-block:: python

      # import os
      # os.environ["HMFAST_DATA_PATH"] = "path/to/hmfast_data"
      import hmfast
      

After these steps, you should be all set to start using ``hmfast``.

