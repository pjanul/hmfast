Installation
============


To use ``hmfast``, you'll need the source code and the auxiliary emulator files.  
The recommended way to get started is as follows:

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/hmfast/hmfast.git

   Then, make sure you pip install the local repository.

   .. code-block:: bash

      pip install /your/path/to/hmfast


2. **Download emulator data files:**

   The easiest way to get started is via Python.

   To download the recommended emulator models (ede-v2) to the default location (``~/hmfast_data``):

   .. code-block:: python

      import hmfast
      hmfast.download_emulators()

   If you wish to store the data elsewhere, simply set the environment variable to a new location:

   .. code-block:: python

      import hmfast, os
      os.environ["HMFAST_DATA_PATH"] = "path/to/hmfast_data"
      hmfast.download_emulators()

   To download all or multiple emulator models at once, use the ``models`` argument:

   .. code-block:: python

      # Download all available models
      hmfast.download_emulators(models="all")

      # Download select models. Can be any of ["lcdm", "mnu", "neff", "wcdm", "ede-v1", "mnu-3states", "ede-v2"]
      hmfast.download_emulators(models=["lcdm", "ede-v2"])

   You may pass a single model name, a list of model names, or ``"all"`` to download all models.


After these steps, you should be all set to start using ``hmfast``.

