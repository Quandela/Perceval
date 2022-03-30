from typing import Type, Union

from .template import Backend
from .cliffords2017 import CliffordClifford2017Backend
from .naive import NaiveBackend
from .slos import SLOSBackend
from .stepper import StepperBackend
from .strawberryfields import SFBackend


class BackendFactory:
    _backends = (NaiveBackend, CliffordClifford2017Backend, SFBackend, SLOSBackend, StepperBackend)

    def get_backend(self,
                    name: Union[str, None] = None) \
            -> Type[Backend]:
        """Returns a simulator backend

        :param name: The name of the simulator
        :return: the backend
        """
        if name is None:
            name = "SLOS"
        for backend in self._backends:
            if backend.name == name:
                return backend
        # TODO: check this exception
        raise ValueError("Unknown backend: %s" % name)

    def list_backend(self):
        return [backend.name for backend in self._backends]
