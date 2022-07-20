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

def source_shape(canvas, **opts):
    r = 10
    color = "lightgray"
    if 'color' in opts:
        color = opts['color']
    canvas.add_mpath(["M", 0, 25, "c", 0, 0, 0, -r, r, -r,
                      "h", 8, "v", 2*r, "h", -8,
                      "c", -r, 0, -r, -r, -r, -r, "z"],
                     stroke="black", stroke_width=1, fill=color)
    if 'name' in opts and opts['name']:
        canvas.add_text((8, 44), text='['+opts['name']+']', size=6, ta="middle", fontstyle="italic")
    if 'content' in opts and opts['content']:
        canvas.add_text((10, 28), text=opts['content'], size=7, ta="middle")


def detector_shape(canvas, **opts):
    r = 10  # Radius of the half-circle
    color = "lightgray"
    if 'color' in opts:
        color = opts['color']
    canvas.add_mpath(["M", 20, 35, "h", -8, "v", -2*r, "h", 8,
                      "c", 0, 0, r, 0, r, r,
                      "c", 0, r, -r, r, -r, r, "z"],
                     stroke="black", stroke_width=1, fill=color)
    if 'name' in opts:
        canvas.add_text((18, 44), text='['+opts['name']+']', size=6, ta="middle", fontstyle="italic")
    if 'content' in opts:
        canvas.add_text((20, 28), text=opts['content'], size=7, ta="middle")
