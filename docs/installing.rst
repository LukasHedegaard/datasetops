Installing
==========

The library can be obtained either as a package through PyPI and Conda, or built from source.

To install through a package manager use:

.. code-block:: bash

   pip install datasetops
   conda install datasetops
   
The package can be installed from source by cloning the git repository:

.. code-block:: bash
   
   git clone https://github.com/LukasHedegaard/datasetops.git

And then running pip install in the root of the folder:

.. code-block:: bash
   
   pip install .

This will install the package in the current environment.

In case you wish to edit the code it may be useful to pass the editable flag to pip:

.. code-block:: bash

   pip install -e .

This makes imports point to the local files instead of the ones installed globally. 
The benefit of this is that pip install does not have to be invoked manually to update files.