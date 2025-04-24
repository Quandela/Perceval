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

from perceval import catalog


def get_pretty_string(s: str):
    out = ''
    for i, c in enumerate(s):
        if i == 0:
            out += c.upper()
            continue
        if s[i-1] == ' ':
            out += c.upper()
            continue
        out += c
    return out


def build_catalog_rst(path: str):
    out = ''
    for key in catalog.list():
        item = catalog[key]
        out += get_pretty_string(item.name) + '\n'
        out += '-'*len(item.name) + '\n\n'
        out += f'Catalog key: ``{item.name}``\n\n'
        out += item.description + '\n\n'

        if item.params_doc:
            out += 'Parameters:\n'
            for param_name, param_descr in item.params_doc.items():
                out += f'    * ``{param_name}``: {param_descr}\n'
            out += '\n'

        out += '.. code-block::\n\n'
        out += '    ' + item.str_repr.replace('\n', '\n    ')+'\n\n'

        if item.see_also:
            out += f'See also: {item.see_also}\n\n'

        if item.article_ref:
            out += f'Scientific article reference: {item.article_ref}\n\n'

    with open(path, 'w', encoding="utf-8") as file:
        file.write(out)
