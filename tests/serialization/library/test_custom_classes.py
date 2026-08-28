# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Note: this file is both an example on how to integrate custom classes to the serialization system and a test on the system itself

from perceval.serialization import Serialization, OutputArchive, InputArchive, PartialRecord, DescriptorString, PreRecorder


def test_data_class():
    # In this scenario, we always serialize the same given members of a class

    class A:
        # Example: everything is directly written in the class
        class_serial_members: list[str] = ["a", "b", "c"]  # Mandatory, name of the attributes that will be serialized
        class_version: int = 42  # Defaults to 0
        class_tag: str = "A_data"  # Defaults to the class name

        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def __eq__(self, other):
            return self.a == other.a and self.b == other.b and self.c == other.c

    Serialization.register_class(A)

    a = A(10, 20, 30)

    ar = OutputArchive()
    Serialization.serialize(a, ar)

    assert ar.to_text() == "pcvlar 0 1 0 A_data 42 3 a 1 b 2 c 3 int 10 int 20 int 30"

    deser = InputArchive.from_text(ar.to_text())
    assert Serialization.deserialize(deser) == a

    class B:
        # Other example: everything is given at registration (i.e. the class is unaware of the Serialization)
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def __eq__(self, other):
                    return self.a == other.a and self.b == other.b

    Serialization.register_class(B, ["a", "b"], tag="B_test", version=420)
    a = B(3.14, ["test"])
    ar = OutputArchive()
    Serialization.serialize(a, ar)

    assert ar.to_text() == "pcvlar 0 1 0 B_test 420 2 a 1 b 2 float 3.14 list 1 3 str 4 test"

    deser = InputArchive.from_text(ar.to_text())
    assert Serialization.deserialize(deser) == a


def test_data_split():
    # In this scenario, we (de)serialize only some members given some conditions.
    # When changing the members of a data_class (see test_data_class),
    # it is good practice to change the version and use it as a dispatcher in a data_split.

    class A:
        # Example: the class knows everything about the serialization

        class_version = 1
        class_tag = "A_data_split"

        def __init__(self, a, b):
            self.a = a
            # self.b = b + 1  # On version 0
            self._b = b  # On version 1

        def __eq__(self, other):
            return self.a == other.a and self._b == other._b

        def class_serial_members_write(self, ar: OutputArchive) -> PartialRecord:
            # Using save_attr makes the equivalent to declaring the class_data_members in a data_class
            return ar.save_attr(self, ["a", "_b"])

        def class_serial_members_read(self, ar: InputArchive, desc: list[tuple[str, int]], version: int):
            # This must fill self from the descriptor
            if version == self.class_version:
                ar.load_attr(self, desc)
            elif version == 0:
                # Note: if this class is using __slots__, it will be impossible to fill it with unknown arguments.
                # It is possible to create an empty object only for loading and filling purpose like this
                # class Temp: pass
                # temp = Temp()
                # ar.load_attr(temp, desc)
                # self.a = temp.a
                # ...

                ar.load_attr(self, desc)
                # Now, self has a member b
                self._b = self.b - 1
                delattr(self, "b")
            else:
                raise RuntimeError(f"Unknown version {version}")

    Serialization.register_class(A)
    a = A(3.14, 2)
    ar = OutputArchive()
    Serialization.serialize(a, ar)

    assert ar.to_text() == f"pcvlar 0 1 0 {A.class_tag} {A.class_version} 2 a 1 _b 2 float 3.14 int 2"

    deser = InputArchive.from_text(ar.to_text())
    assert Serialization.deserialize(deser) == a

    # Now, suppose we have the serialization from version 0
    old_serialized = f"pcvlar 0 1 0 {A.class_tag} 0 2 a 1 b 2 float 3.14 int 3"
    deser = InputArchive.from_text(old_serialized)
    assert Serialization.deserialize(deser) == a


    # Now, we can do the same by giving everything externally
    class B:
        def __init__(self, a, b):
            self.a = a
            # self.b = b + 1  # On version 0
            self._b = b  # On version 1

        def __eq__(self, other):
            return self.a == other.a and self._b == other._b

    def read_b(b: B, ar: InputArchive, desc: list[tuple[str, int]], version: int):
        # For demonstration purpose, we achieve the same result but differently
        for name, idx in desc:
            value = ar.create(idx)
            if name == "b":
                name = "_b"
                value -= 1

            setattr(b, name, value)

    B_tag = "B_data_split"
    Serialization.register_class(B,
                                 class_serial_members_write=lambda b, ar: ar.save_attr(b, ["a", "_b"]),
                                 class_serial_members_read=read_b,
                                 tag = B_tag,
                                 version = 1)

    a = B(3.14, 2)
    ar = OutputArchive()
    Serialization.serialize(a, ar)

    assert ar.to_text() == f"pcvlar 0 1 0 {B_tag} 1 2 a 1 _b 2 float 3.14 int 2"

    deser = InputArchive.from_text(ar.to_text())
    assert Serialization.deserialize(deser) == a

    # Now, suppose we have the serialization from version 0
    old_serialized = f"pcvlar 0 1 0 {B_tag} 0 2 a 1 b 2 float 3.14 int 3"
    deser = InputArchive.from_text(old_serialized)
    assert Serialization.deserialize(deser) == a


def test_class():
    # In this scenario, we want to use a custom serialization process.
    # This requires the use of a DescriptorType, which is an internal class able to write one basic python class
    # If needed, a custom DescriptorType can be written, but in most cases, this shouldn't be necessary

    class A:
        # First possibility: declare everything internally
        class_tag = "A_class"
        class_serializer_type = DescriptorString  # Needed in this scenario

        def __init__(self, a: str):
            self.a = a

        def __eq__(self, other):
            return self.a == other.a

        def class_write_custom(self, ar: OutputArchive) -> PartialRecord:
            # Note: if a class version is needed, it must be added by hand
            return self.class_serializer_type(self.a), []  # The list is the list of all children that would need to be recorded

        @staticmethod
        def class_read_custom(ar: InputArchive, desc: DescriptorString, pre_recorder: PreRecorder):
            # Use the PreRecorder to register a new instance of this object before creating the members in the archive
            # if the members could point to this new object
            return A(desc.value)


    Serialization.register_class(A)
    a = A("test")
    ar = OutputArchive()
    Serialization.serialize(a, ar)

    assert ar.to_text() == f"pcvlar 0 1 0 {A.class_tag} 4 test"

    deser = InputArchive.from_text(ar.to_text())
    assert Serialization.deserialize(deser) == a

    # Again, everything can be externally given
    class B:
        def __init__(self, a: str):
            self.a = a

        def __eq__(self, other):
            return self.a == other.a

    Serialization.register_class(B,
                                 class_write_custom=lambda b, ar: (DescriptorString(b.a), []),
                                 class_read_custom=lambda ar, desc, pre_recorder: B(desc.value),
                                 descriptor_type=DescriptorString,
                                 tag="B_class"
                                 )

    a = B("test b")
    ar = OutputArchive()
    Serialization.serialize(a, ar)
    assert ar.to_text() == f"pcvlar 0 1 0 B_class 6 test b"

    deser = InputArchive.from_text(ar.to_text())
    assert Serialization.deserialize(deser) == a
