Registering new classes
=======================

A class must be registered before one of its instances can be added to an archive. Registration associates the exact
Python class with a stable text tag and the functions used to save and restore it.

Registration is global and should normally happen once, at module import time, after the class declaration. Tags must
be unique. The default tag is the class name, but an explicit, stable tag is recommended when similarly named classes
can coexist.

.. important::
   Registration uses the exact type of an object. Registering a base class does not register its subclasses. Register
   each concrete subclass separately so that deserialization always restores the intended type.

.. important::
   Registration is needed at both ends of the serialization. Sending a serialized format to someone else will only work
   if they have all the required classes in their registry.
   Also, beware of tag uniqueness for compatibility between frameworks.

Fixed data members
^^^^^^^^^^^^^^^^^^

For a class that can always be reconstructed by assigning the same attributes, list those attributes when registering
the class:

>>> from perceval.serialization import Serialization
>>>
>>> class Point:
...     def __init__(self, x, y):
...         self.x = x
...         self.y = y
...
>>> Serialization.register_class(Point, ["x", "y"], tag="Point")

The values of ``x`` and ``y`` must themselves have registered classes. During deserialization, the system creates an
uninitialized ``Point`` with ``Point.__new__`` and assigns the saved attributes; it does not call ``Point.__init__``.

A class can declare the same information itself:

.. code-block:: python

   class Point:
       class_serial_members = ["x", "y"]
       class_tag = "Point"
       class_version = 0

       def __init__(self, x, y):
           self.x = x
           self.y = y


   Serialization.register_class(Point)

``class_tag`` defaults to the class name and ``class_version`` defaults to ``0``.

Separate write and read functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use separate functions when some runtime state must be omitted, derived values must be rebuilt, or older archive
versions require migration. The writer returns the result of
:meth:`~perceval.serialization.OutputArchive.save_attr`. The reader fills the instance that the archive has already
allocated:

.. code-block:: python

   from perceval.serialization import InputArchive, Serialization


   class ServiceConnection:
       def __init__(self, endpoint, token):
           self.endpoint = endpoint
           self.token = token
           self.client = build_client(endpoint, token)


   def save_connection(connection, archive):
       # The live client is deliberately omitted.
       return archive.save_attr(connection, ["endpoint", "token"])


   def load_connection(connection, archive, members, version):
       archive.load_attr(connection, members)
       connection.client = build_client(connection.endpoint, connection.token)


   Serialization.register_class(
       ServiceConnection,
       class_serial_members_write=save_connection,
       class_serial_members_read=load_connection,
       tag="ServiceConnection",
       version=0,
   )

The ``members`` argument received by the reader is a list of ``(attribute_name, archive_index)`` pairs. Calling
``archive.load_attr(instance, members)`` restores them directly. For transformations or migrations, use
``archive.create(index)`` to retrieve individual values before assigning them.

These methods can also be written directly in the class under the names ``class_serial_members_write`` and
``class_serial_members_read``.

Versioning serialized data
^^^^^^^^^^^^^^^^^^^^^^^^^^

The class version is stored in every class descriptor. Increment it whenever the serialized representation changes,
then dispatch on ``version`` in the reader to support existing archives:

.. code-block:: python

   def load_record(record, archive, members, version):
       values = {name: archive.create(index) for name, index in members}

       if version == 0:
           record.new_name = values["old_name"]
       elif version == 1:
           record.new_name = values["new_name"]
       else:
           raise RuntimeError(f"Unsupported Record version {version}")

When backward compatibility is not possible, the reader should explicitly reject unknown versions instead of
silently constructing an incomplete object.

Custom representations
^^^^^^^^^^^^^^^^^^^^^^^

For a class with a canonical compact representation, register custom writer and reader functions together with the
descriptor type that stores it. This example stores a ``Label`` as a single string:

.. code-block:: python

   from perceval.serialization import DescriptorString, Serialization


   class Label:
       def __init__(self, value):
           self.value = value


   def write_label(label, archive):
       return DescriptorString(label.value), []


   def read_label(archive, descriptor, pre_recorder):
       return Label(descriptor.value)


   Serialization.register_class(
       Label,
       class_write_custom=write_label,
       class_read_custom=read_label,
       descriptor_type=DescriptorString,
       tag="Label",
   )

The writer returns a descriptor and a list of child objects that the archive must visit. The example has no children,
so the list is empty. If a custom reader creates an object whose children may refer back to it, call
``pre_recorder(instance)`` before creating those children so cyclic references can be resolved.

Custom descriptors are an advanced extension point. Prefer fixed members or split read/write functions when an
existing archive descriptor cannot express the complete representation on its own.

These methods can also be written directly in the class under the names ``class_write_custom`` and
``class_read_custom``. The ``descriptor_type`` can be in the class under the name ``class_serializer_type``.

Choosing a registration strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Use **fixed data members** for stable classes that only need attribute assignment.
* Use **separate write and read functions** to omit runtime-only state, rebuild resources, or migrate versions.
* Use a **custom representation** for an existing canonical form, such as a string or binary representation.

All three forms are available with a single method: :meth:`~perceval.serialization.Serialization.register_class`.

API reference
^^^^^^^^^^^^^

.. automethod:: perceval.serialization.Serialization.register_class

.. automethod:: perceval.serialization.Serialization.register_data

.. automethod:: perceval.serialization.Serialization.register_data_split

.. automethod:: perceval.serialization.Serialization.register_custom_class
