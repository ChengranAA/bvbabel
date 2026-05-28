"""Core infrastructure for typed binary format reading and writing.

Provides a declarative field-descriptor system that allows BrainVoyager
file formats to be defined as typed Python classes.  A ``BinaryFormat``
subclass declares fields as class-level descriptors; the base class
auto-collects them in declaration order and provides generic ``read()``
and ``write()`` methods that iterate over the fields, handling struct
packing/unpacking, conditional presence, and data-axis transforms.

Hooks
-----
Subclasses may override ``_post_read()`` and ``_pre_write()`` to perform
custom reshaping or cross-field fixups that cannot be expressed
declaratively.  ``_post_read`` runs after all fields are read (before
``read()`` returns).  ``_pre_write`` runs before fields are written
(inside ``write()``).

Usage
-----
    class MyFormat(BinaryFormat):
        version = Field("<H")
        name    = StringField()
        _data   = DataField("<f", shape_fields=("dim_x", "dim_y"))
        # ...

    obj = MyFormat.read("file.bin")
    obj.write("output.bin")

    # Header-only (skip the large data block)
    meta = MyFormat.read("file.bin", load_data=False)

Dynamic dispatch
----------------
    import bvbabel._binary_format as bf
    obj = bf.load("some_file.vmr")           # → VMR instance
    obj = bf.load("some_file.vmr", load_data=False)
"""

import struct
import os
import numpy as np

# ---------------------------------------------------------------------------
# Field descriptors
# ---------------------------------------------------------------------------


class Field:
    """Descriptor for a single binary field backed by ``struct``.

    Parameters
    ----------
    fmt : str
        ``struct`` format character (e.g. ``"<H"``, ``"<i"``, ``"<f"``).
    condition : callable, optional
        Callable receiving the instance; the field is read/written only
        when the callable returns ``True``.  Used for version-gated fields.
    default : any, optional
        Default value when the field has not been set.
    """

    def __init__(self, fmt="", condition=None, default=0):
        self.fmt = fmt
        self.condition = condition
        self.default = default
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance._values.get(self.name, self.default)

    def __set__(self, instance, value):
        instance._values[self.name] = value

    def _should_process(self, instance):
        if self.condition is None:
            return True
        return self.condition(instance)

    def read(self, f, instance, load_data=True):
        if not self._should_process(instance):
            return
        size = struct.calcsize(self.fmt)
        data, = struct.unpack(self.fmt, f.read(size))
        instance._values[self.name] = data

    def write(self, f, instance):
        if not self._should_process(instance):
            return
        value = instance._values.get(self.name, self.default)
        f.write(struct.pack(self.fmt, value))


class StringField(Field):
    """Variable-length null-terminated string field."""

    def __init__(self, condition=None, default=""):
        super().__init__(fmt=None, condition=condition, default=default)

    def read(self, f, instance, load_data=True):
        if not self._should_process(instance):
            return
        text = ""
        data = f.read(1)
        while data != b"\x00":
            text += data.decode("utf-8", "ignore")
            data = f.read(1)
        instance._values[self.name] = text

    def write(self, f, instance):
        if not self._should_process(instance):
            return
        text = instance._values.get(self.name, self.default)
        for ch in text:
            f.write(struct.pack("<s", ch.encode("utf-8")))
        f.write(b"\x00")


class RGBField(Field):
    """3-byte RGB colour field stored as a ``numpy.ubyte`` array of length 3."""

    def __init__(self, condition=None):
        super().__init__(fmt=None, condition=condition, default=None)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        val = instance._values.get(self.name)
        if val is None:
            val = np.zeros(3, dtype=np.ubyte)
        return val

    def read(self, f, instance, load_data=True):
        if not self._should_process(instance):
            return
        rgb = np.zeros(3, dtype=np.ubyte)
        for i in range(3):
            data, = struct.unpack("<B", f.read(1))
            rgb[i] = data
        instance._values[self.name] = rgb

    def write(self, f, instance):
        if not self._should_process(instance):
            return
        rgb = instance._values.get(self.name, np.zeros(3, dtype=np.ubyte))
        for i in range(3):
            f.write(struct.pack("<B", int(rgb[i])))


class DataField:
    """Descriptor for a raw numpy data block embedded in the binary stream.

    When *load_data* is ``False``, the field seeks past the raw bytes
    and sets the value to ``None`` — enabling fast header-only reads.

    Parameters
    ----------
    dtype : str or numpy.dtype
        Data type for the raw bytes (e.g. ``"<B"``, ``"<f"``).
    shape_fields : tuple of str
        Names of fields whose values determine the array dimensions.
    transform / inverse_transform : callable, optional
        Called on the numpy array after reading / before writing.
    """

    def __init__(self, dtype, shape_fields, transform=None,
                 inverse_transform=None):
        self.dtype = np.dtype(dtype)
        self.shape_fields = shape_fields
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance._values.get(self.name)

    def __set__(self, instance, value):
        instance._values[self.name] = value

    def _byte_size(self, instance):
        shape = tuple(getattr(instance, n) for n in self.shape_fields)
        return int(np.prod(shape)) * self.dtype.itemsize

    def read(self, f, instance, load_data=True):
        if not load_data:
            f.seek(self._byte_size(instance), os.SEEK_CUR)
            instance._values[self.name] = None
            return
        shape = tuple(getattr(instance, n) for n in self.shape_fields)
        count = int(np.prod(shape))
        data = np.fromfile(f, dtype=self.dtype, count=count)
        data = data.reshape(shape)
        if self.transform is not None:
            data = self.transform(data)
        instance._values[self.name] = data

    def write(self, f, instance):
        data = instance._values.get(self.name)
        if data is None:
            return
        if self.inverse_transform is not None:
            data = self.inverse_transform(data)
        f.write(data.astype(self.dtype).tobytes(order="C"))


class SubRecordListField:
    """Descriptor for a list of dict-based sub-records with callbacks."""

    def __init__(self, count_field, read_record, write_record,
                 condition=None):
        self.count_field = count_field
        self._read_record = read_record
        self._write_record = write_record
        self.condition = condition
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance._values.get(self.name, [])

    def __set__(self, instance, value):
        instance._values[self.name] = value

    def _should_process(self, instance):
        if self.condition is not None and not self.condition(instance):
            return False
        count = getattr(instance, self.count_field, 0)
        return count > 0

    def read(self, f, instance, load_data=True):
        if not self._should_process(instance):
            instance._values[self.name] = []
            return
        count = getattr(instance, self.count_field)
        records = []
        for _ in range(count):
            records.append(self._read_record(f))
        instance._values[self.name] = records

    def write(self, f, instance):
        if not self._should_process(instance):
            return
        records = instance._values.get(self.name, [])
        for rec in records:
            self._write_record(f, rec)


class TypedSubRecordListField:
    """Descriptor for a list of typed ``BinaryFormat`` sub-records.

    Each sub-record is a ``BinaryFormat`` subclass whose own fields are
    iterated in order.  ``post_record_read`` / ``post_record_write``
    callbacks fire after each record, enabling trailing sections that
    reference parent-context fields (e.g. time courses interleaved with
    statistical maps).

    Parameters
    ----------
    count_field : str
        Name of the field storing the number of records.
    record_cls : BinaryFormat subclass
    post_record_read : callable, optional
        ``post_record_read(f, parent, record)``
    post_record_write : callable, optional
        ``post_record_write(f, parent, record)``
    condition : callable, optional
    """

    def __init__(self, count_field, record_cls,
                 post_record_read=None, post_record_write=None,
                 condition=None):
        self.count_field = count_field
        self.record_cls = record_cls
        self._post_record_read = post_record_read
        self._post_record_write = post_record_write
        self.condition = condition
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance._values.get(self.name, [])

    def __set__(self, instance, value):
        instance._values[self.name] = value

    def _should_process(self, instance):
        if self.condition is not None and not self.condition(instance):
            return False
        return getattr(instance, self.count_field, 0) > 0

    def read(self, f, instance, load_data=True):
        if not self._should_process(instance):
            instance._values[self.name] = []
            return
        count = getattr(instance, self.count_field)
        records = []
        for _ in range(count):
            rec = self.record_cls()
            for field in self.record_cls._fields:
                field.read(f, rec, load_data=load_data)
            records.append(rec)
            if self._post_record_read is not None:
                self._post_record_read(f, instance, rec)
        instance._values[self.name] = records

    def write(self, f, instance):
        if not self._should_process(instance):
            return
        records = instance._values.get(self.name, [])
        for rec in records:
            for field in self.record_cls._fields:
                field.write(f, rec)
            if self._post_record_write is not None:
                self._post_record_write(f, instance, rec)



# ---------------------------------------------------------------------------
# Section and ObjectField
# ---------------------------------------------------------------------------


class Section:
    """Lightweight container with Field descriptor support."""

    def __init__(self, **kwargs):
        self._values = {}
        for key, val in kwargs.items():
            self._values[key] = val


class ObjectField:
    """Descriptor storing a sub-object with no-op binary read/write."""

    def __init__(self, default_factory=None):
        self.default_factory = default_factory
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        val = instance._values.get(self.name)
        if val is None and self.default_factory is not None:
            val = self.default_factory()
            instance._values[self.name] = val
        return val

    def __set__(self, instance, value):
        instance._values[self.name] = value

    def read(self, f, instance, load_data=True):
        pass

    def write(self, f, instance):
        pass

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BinaryFormat:
    """Base class for typed binary-format objects.

    Hooks
    -----
    ``_post_read()``
        Called after all fields are read.  Override to reshape data arrays
        or perform cross-field fixups.
    ``_pre_write()``
        Called before fields are written.  Override to flatten data arrays
        back to on-disk layout.
    """

    _fields: list = []

    # -- Subclass initialisation -----------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        fields = []
        seen = set()

        for base in reversed(cls.__mro__):
            if base is cls or base is object:
                continue
            for f in getattr(base, "_fields", []):
                if f.name is not None and f.name not in seen:
                    fields.append(f)
                    seen.add(f.name)

        _field_types = (Field, DataField, SubRecordListField,
                        TypedSubRecordListField, ObjectField)
        for name in cls.__dict__:
            value = cls.__dict__[name]
            if isinstance(value, _field_types):
                if value.name is None:
                    value.name = name
                if value.name in seen:
                    for i, f in enumerate(fields):
                        if f.name == value.name:
                            fields[i] = value
                            break
                else:
                    fields.append(value)
                    seen.add(value.name)

        cls._fields = fields

    # -- Instance --------------------------------------------------------

    def __init__(self, **kwargs):
        self._values = {}
        for key, val in kwargs.items():
            self._values[key] = val

    # -- I/O -------------------------------------------------------------

    @classmethod
    def read(cls, filename, load_data=True):
        """Read *filename* and return a populated instance."""
        instance = cls()
        with open(filename, "rb") as f:
            for field in cls._fields:
                field.read(f, instance, load_data=load_data)
        instance._post_read()
        return instance

    def write(self, filename):
        """Write this instance to *filename* in binary format."""
        self._pre_write()
        with open(filename, "wb") as f:
            for field in self._fields:
                field.write(f, self)
        self._post_write()

    # -- Hooks (override in subclasses) ----------------------------------

    def _post_read(self):
        """Called after all fields have been read from disk."""

    def _pre_write(self):
        """Called before fields are written to disk."""

    def _post_write(self):
        """Called after all fields have been written.

        By default delegates to ``_post_read()`` so that subclasses
        that reshape data in ``_pre_write`` get their logical shape
        restored automatically.
        """
        self._post_read()

    # -- Dict helpers ----------------------------------------------------

    def to_dict(self):
        """Return field values as a plain ``dict`` (shallow copy)."""
        return dict(self._values)

    @classmethod
    def from_dict(cls, d):
        """Create an instance from a plain ``dict`` of field values."""
        return cls(**d)


# ---------------------------------------------------------------------------
# Dynamic-dispatch registry
# ---------------------------------------------------------------------------

_registry = {}


def register_format(extensions):
    """Class decorator that registers a format for one or more extensions."""
    if isinstance(extensions, str):
        extensions = [extensions]

    def decorator(cls):
        for ext in extensions:
            _registry[ext.lower()] = cls
        return cls

    return decorator


def load(filename, load_data=True):
    """Open *filename* and return the appropriate typed-format instance."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _registry:
        raise ValueError(
            f"No format registered for extension {ext!r}. "
            f"Known extensions: {list(_registry)}"
        )
    return _registry[ext].read(filename, load_data=load_data)
