PersistentData
==============

:code:`PersistentData` is a class that allows to save and store data across perceval launches.
Most importantly, it is used to save :ref:`RemoteConfig`, :ref:`JobGroup`...

.. warning::
  The folder created by :code:`PersistentData` is never emptied automatically.
  This means that using features that make use of :code:`PersistentData` may use a lot of disk after many uses.

Usage example:

>>> import perceval as pcvl
>>> pdata = pcvl.PersistentData()
>>> pdata.write_file("my_file.txt", "my_data", pcvl.FileFormat.TEXT)
>>> print(pdata.read_file("my_file.txt", pcvl.FileFormat.TEXT))
my_data
>>> pdata.delete_file("my_file.txt")

.. note::
  The default folder is created inside the user folder, so the persistent data are not shared between users.

.. autoclass:: perceval.utils.persistent_data.PersistentData
  :members:

.. autoenum:: perceval.utils._enums.FileFormat
