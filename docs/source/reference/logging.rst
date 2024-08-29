Logging
=======

To log with Perceval you can either use our logger or the python one. By default, our logger will be used.

To have the perceval message log with the python logger, use this method:

.. code-block:: python

    import perceval
    perceval.utils.use_python_logger()

To use back the perceval logger use this method:

.. code-block:: python

    import perceval
    perceval.utils.use_perceval_logger()

.. note:: If you use the python logger, use directly the module logging of python to configure it (except for channel configuration)

This module defines functions and classes which implement a flexible event logging system for any Perceval script.

It is build to work similarly as the python logging module.

Logger
------

The logger is instantiate at perceval initialization.

In order to use it you'll just need to import it:

.. code-block:: python

    from perceval.utils import logger

To log message, you can use it the same way as the python logger:

.. code-block:: python

    from perceval.utils import logger
    logger.info('I log something as info')


Saving log in file
^^^^^^^^^^^^^^^^^^

By default the log are not save in a file.
If this feature is activated, the log file will be in the perceval persistent data folder and the path of the file will be printed at the beginning of your perceval script.

If order to activate / deactivate the logging in a file you can use the following methods:

.. code-block:: python

    from perceval.utils import logger
    logger.enable_file()
    logger.disable_file()

Levels
------

You can use the logger to log message at different level. Each level represent a different type of message.

The level are listed by ascending important in the following table.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :width: 100%
   :align: center

   * - Log level
     - Perceval call
     - Usage
   * - DEBUG
     - ``logger.debug``
     - Detailed information, typically of interest only when diagnosing problems.
   * - INFO
     - ``logger.info``
     - Confirmation that things are working as expected.
   * - WARNING
     - ``logger.warn``
     - An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
   * - ERROR
     - ``logger.error``
     - Due to a more serious problem, the software has not been able to perform some function.
   * - CRITICAL
     - ``logger.critical``
     - A serious error, indicating that the program itself may be unable to continue running.

Example
^^^^^^^

.. code-block:: python

    from perceval.utils import logger
    logger.info('I log something as info')
    logger.critical('I log something as critical')

Channels
--------

You can also log in a specific channel. A channel is like a category.
Each channel can have its own configuration, which means each channel can have a different level.
If the channel is not specified, the message will be log in the ``user`` channel.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :width: 100%
   :align: center

   * - Channel
     - Default level
     - Usage
   * - ``general``
     - off
     - General info: Deprecated, Gate not to use, weird simulation results
   * - ``resources``
     - off
     - Info about how are use our backends or remote platform GPU use (exqalibur)
   * - ``user``
     - warning
     - Channel to use as a Perceval user, this will have to be clearly stated in the documentation

Example
^^^^^^^

.. code-block:: python

    from perceval.utils import logger, logging
    logger.info('I log something as info in channel general', logging.channel.general)

To set a level for a channel you can use the following method:

.. code-block:: python

    from perceval.utils import logger
    logger.set_level(level, channel)

Example
^^^^^^^

.. code-block:: python

    from perceval.utils import logger, logging
    logger.set_level(logging.level.info, logging.channel.general)

Logger configuration
--------------------

For logging to be useful, it needs to be configured, meaning setting the levels for each channel and if log are saved in a file.
Setting a level for a channel means that any log with a less important level will not be displayed/save.

In most cases, only the user & general channel needs to be so configured, since all relevant messages will be log here.

Example
^^^^^^^

.. code-block:: python

    from perceval.utils import logger, logging
    logger.enable_file()
    logger.set_level(logging.level.info, logging.channel.resources)
    logger.set_level(logging.level.err, logging.channel.general)

.. note:: The logger configuration can also be stored in the persistent data so you don't have to configure the logger each time you use perceval.

In order to configure it you have use the class LoggerConfig.

.. automodule:: perceval.utils.logging.config
   :members:

After configuring your LoggerConfig, you can apply it to the current logger:

.. code-block:: python

    from perceval.utils import logger, logging
    logger_config = logging.LoggerConfig()
    logger_config.enable_file()
    logger_config.set_level(logging.level.info, logging.channel.user)
    logger.apply_config(logger_config)

Log format
----------

In the console the log will appear with the format:

[log_level] message

In the file, the log will be save to the format:

[yyyy-mm-dd HH:MM:SS.fff]channel_first_letter[level_first_letter] message

Log exceptions
--------------
If the general channel level is at least on critical and save in file is enable, uncaught exception will be logged
