utils.logging
=============

To log with Perceval you can either use a built-in Perceval logger or the python one. By default, our logger will be used.

To log the perceval messages with the python logger, use this method:

.. code-block:: python

    from perceval.utils import use_python_logger
    use_python_logger()

To switch back to using perceval logger, use:

.. code-block:: python

    from perceval.utils import use_perceval_logger
    use_perceval_logger()

.. note:: If you use the python logger, use directly the module logging of python to configure it (except for channel configuration)

This module defines functions and classes which implement a flexible event logging system for any Perceval script.

It is build to work similarly as the python logging module.

Logger
------

A logger instance is created the first time Perceval is imported.

In order to use it you'll just need to import it:

.. code-block:: python

    from perceval.utils import get_logger

To log a message, you can use it the same way as the python logger:

.. code-block:: python

    from perceval.utils import get_logger
    get_logger().info('I log something as info')
    # or
    logger = get_logger()
    logger.info('1st message')
    logger.info('2nd message')


Saving log to file
^^^^^^^^^^^^^^^^^^

By default the log are not saved in a file.

If order to enable / disable the writing of logs in a file, use the following methods:

.. code-block:: python

    from perceval.utils import get_logger
    get_logger().enable_file()
    get_logger().disable_file()

When this feature is enabled, log files are written to Perceval persistent data folder and the path of the file will be printed when your script starts writing inside it.

Levels
------

You can use the logger to log message at different level. Each level represent a different type of message.

The level are listed by ascending order of importance in the following table.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :width: 100%
   :align: center

   * - Log level
     - Perceval call
     - Usage
   * - DEBUG
     - ``level.debug``
     - Detailed technical information, typically of interest only when diagnosing problems.
   * - INFO
     - ``level.info``
     - Confirmation of things working as expected.
   * - WARNING
     - ``level.warn``
     - An indication that something unexpected happened, or indicative of some problem in the near future. The software is still working as expected.
   * - ERROR
     - ``level.err``
     - Due to a more serious problem, the software has not been able to perform some function.
   * - CRITICAL
     - ``level.critical``
     - A serious error, indicating that the program itself is unable to continue running normally or has crashed.

Example
^^^^^^^

.. code-block:: python

    from perceval.utils import get_logger
    get_logger().info('I log something as info')
    get_logger().critical('I log something as critical')

Channels
--------

You can also log in a specific channel. A channel is like a category.
Each channel can have its own configuration, which means each channel can have a different level.
If the channel is not specified, the message is logged in the ``user`` channel.

.. note:: If you are a Perceval user, you should only write log in the channel ``user``.

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
     - General info: Technical info, track the usage of features
   * - ``resources``
     - off
     - Usage info about our backends or remote platform GPU (exqalibur)
   * - ``user``
     - warning
     - Channel to use as a Perceval user & warnings (such as deprecated methods or arguments)

Example
^^^^^^^

.. code-block:: python

    from perceval.utils.logging import get_logger, channel
    get_logger().info('I log something as info in channel user', channel.user)

To set a level for a channel, use the following method:

.. code-block:: python

    from perceval.utils import get_logger
    get_logger().set_level(level, channel)

Example
^^^^^^^

.. code-block:: python

    from perceval.utils.logging import get_logger, level, channel
    get_logger().set_level(level.info, channel.general)

Logger configuration
--------------------

For logging to be useful, it needs to be configured, meaning setting the levels for each channel and if log are saved in a file.
Setting a level for a channel means that any log with a less important level will not be displayed/saved.

In most cases, only the user & general channel needs to be so configured, since all relevant messages will be logged here.

Example
^^^^^^^

.. code-block:: python

    from perceval.utils.logging import get_logger, channel, level
    logger = get_logger()
    logger.enable_file()
    logger.set_level(level.info, channel.resources)
    logger.set_level(level.err, channel.general)

.. note:: The logger configuration can be saved on your hard drive so you don't have to configure the logger each time you use perceval. When saved, it is written to a file in Perceval persistent data folder.

In order to configure it you have to use the :class:`LoggerConfig`.

.. automodule:: perceval.utils.logging.config
   :members:

After configuring your LoggerConfig, you can apply it to the current logger:

.. code-block:: python

    from perceval.utils.logging import get_logger, LoggerConfig, level, channel
    logger_config = LoggerConfig()
    logger_config.enable_file()
    logger_config.set_level(level.info, channel.user)
    get_logger().apply_config(logger_config)

Log format
----------

On the console the log will appear with the format:

[log_level] message

In the file, the log will be save to the format:

[yyyy-mm-dd HH:MM:SS.fff]channel_first_letter[level_first_letter] message

Log exceptions
--------------

If the general channel level is at least on critical and save in file is enable, uncaught exception will be logged and
saved on disk with their callstack.
