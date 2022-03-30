from abc import ABC, abstractmethod
from .source import Source


class PortArray:
    def __init__(self, processor, nport, input=True):
        self._ports = [None] * nport
        self._input = input
        self._processor = processor

    def __setitem__(self, idx, value):
        assert isinstance(idx, int) and idx >= 0 and idx < len(self._ports), "invalid port index"
        if self._input:
            assert isinstance(value, InPort), "invalid port assignment"
        else:
            assert isinstance(value, OutPort), "invalid port assignment"

        assert self._ports[idx] is None, "duplicate port definition"
        self._ports[idx] = value
        value.set_processor(self._processor)

    def __getitem__(self, idx):
        assert self._ports[idx] is not None
        return self._ports[idx]


class InPort(ABC):
    def __init__(self, connect):
        self._connect = connect

    def set_processor(self, p):
        self._processor = p


class InBinaryPort(InPort):
    def __init__(self, source_port_id, connect):
        self._source_port_id = source_port_id
        super().__init__(connect=connect)


class InOpticalPort(InPort):
    def __init__(self, source, connect=None):
        self._source = source
        assert isinstance(source, Source)
        super().__init__(connect)


class InQBitPort(InPort):
    def __init__(self, source, connect):
        self._source = source
        assert isinstance(source, OutQBitPort)
        assert connect is not None
        assert isinstance(connect, tuple) and len(connect) == 2
        super().__init__(connect)


class OutPort(ABC):
    def __init__(self, connect):
        self._connect = connect

    def set_processor(self, p):
        self._processor = p


class OutOpticalPort(OutPort, Source):
    def __init__(self, connect):
        super().__init__(connect)


class OutQBitPort(OutPort):
    def __init__(self, connect):
        assert isinstance(connect, tuple) and len(connect) == 2
        super().__init__(connect)


class OutCounterPort(OutPort):
    def __init__(self, connect):
        assert isinstance(connect, int)
        super().__init__(connect)
