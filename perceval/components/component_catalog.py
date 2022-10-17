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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from abc import ABC, abstractmethod
from enum import Enum
import importlib


class Catalog:
    def __init__(self, path: str):
        self._items = {}
        if path:
            self.add_path(path)

    def add_path(self, path):
        module = importlib.import_module(path)
        if 'catalog' in dir(module):
            sub_catalog = getattr(module, 'catalog')
            self._add_sub_catalog(sub_catalog)
        else:
            warnings.warn(f"No sub catalog found at path {path}", category=ImportWarning)

    def _add_sub_catalog(self, catalog):
        for cls in catalog:
            obj = cls()
            if isinstance(obj, CatalogItem):
                self._items[obj.name] = obj

    def list(self):
        return list(self._items.keys())

    def __contains__(self, item):
        return item in self._items

    def __getitem__(self, item_name: str):
        return self._items[item_name]


class AsType(Enum):
    CIRCUIT = 0
    PROCESSOR = 1


class CatalogItem(ABC):
    article_ref = None
    description = None
    str_repr = None
    see_also = None

    def __init__(self, name: str):
        self._name = name
        self._default_opts = {
            'type': AsType.PROCESSOR,
            'backend': 'SLOS'
        }
        self._reset_opts()

    def _reset_opts(self):
        self._build_opts = self._default_opts.copy()

    def as_circuit(self):
        self._build_opts['type'] = AsType.CIRCUIT
        return self

    def as_processor(self, backend_name: str = None):
        self._build_opts['type'] = AsType.PROCESSOR
        if backend_name is not None:
            self._build_opts['backend'] = backend_name
        return self

    def _opt(self, key):
        if key in self._build_opts:
            return self._build_opts[key]
        return self._default_opts[key] if key in self._default_opts else None

    @property
    def name(self):
        return self._name

    @property
    def doc(self):
        content = ''
        if self.description:
            content += f'\n{self.description}\n'
        if self.article_ref:
            content += f'\nScientific article reference: {self.article_ref}\n'
        if self.str_repr:
            content += f'\nSchema:\n{self.str_repr}\n'
        if self.see_also:
            content += f'\nSee also: {self.see_also}\n'
        if content == '':
            content = 'None'
        title = f'{self._name} documentation\n'.upper()
        title += '-' * len(title) + '\n'
        return title + content

    @abstractmethod
    def build(self):
        pass
