Installation
============


To use ``hmfast``, you'll need the source code and the auxiliary emulator files.  
The recommended way to get started is as follows:

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/hmfast/hmfast.git


2. **Download emulator data files:**

   The easiest way to get started is via Python.

   To download the recommended emulator models (ede-v2) to the default location (``~/hmfast_data``):

   .. code-block:: python

      import hmfast
      hmfast.download_emulators()

   To use a different directory for storing emulator data files, use the ``target_dir`` argument:

   .. code-block:: python

      hmfast.download_emulators(target_dir="/custom/path/for/hmfast_data")

   To download all or multiple emulator models at once, use the ``models`` argument:

   .. code-block:: python

      # Download all available models
      hmfast.download_emulators(models="all")

      # Download several specific models. Can be any of ["lcdm", "mnu", "neff", "wcdm", "ede-v1", "mnu-3states", "ede-v2"]
      hmfast.download_emulators(models=["lcdm", "ede-v2"])

   You may pass a single model name, a list of model names, or ``"all"`` to download all models.


After these steps, you should be all set to start using ``hmfast``.

