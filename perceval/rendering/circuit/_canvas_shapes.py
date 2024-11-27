
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

class ShapeFactory:

    pr_mpath = ["M", 18, 27,
                "c", 0.107, 0.131, 0.280, 0.131, 0.387, 0,
                "l", 2.305, -2.821,
                "c", 0.107, -0.131, 0.057, -0.237, -0.112, -0.237,
                "h", -1.22,
                "c", -0.169, 0, -0.284, -0.135, -0.247, -0.300,
                "c", 0.629, -2.866, 3.187, -5.018, 6.240, -5.018,
                "c", 3.524, 0, 6.39, 2.867, 6.390, 6.3902,
                "c", 0, 3.523, -2.866, 6.39, -6.390, 6.390,
                "c", -0.422, 0, -0.765, 0.342, -0.765, 0.765,
                "s", 0.342, 0.765, 0.765, 0.765,
                "c", 4.367, 0, 7.92, -3.552, 7.920, -7.920,
                "c", 0, -4.367, -3.552, -7.920, -7.920, -7.920,
                "c", -3.898, 0, -7.146, 2.832, -7.799, 6.546,
                "c", -0.029, 0.166, -0.184, 0.302, -0.353, 0.302,
                "h", -1.201,
                "c", -0.169, 0, -0.219, 0.106, -0.112, 0.237,
                "z"]

    @staticmethod
    def half_circle_port_in(radius: float):
        return ["M", 7, 25,
                "c", 0, 0, 0, -radius, radius, -radius,
                "h", 8,
                "v", 2 * radius,
                "h", -8,
                "c", -radius, 0, -radius, -radius, -radius, -radius,
                "z"]

    @staticmethod
    def half_circle_port_out(radius: float, x_offset: float = 8):
        return ["M", x_offset, 35,
                "h", -8,
                "v", -2 * radius,
                "h", 8,
                "c", 0, 0, radius, 0, radius, radius,
                "c", 0, radius, -radius, radius, -radius, radius,
                "z"]

    @staticmethod
    def triangle_port_out(size: float, x_offset: float = 0):
        return ["M", x_offset, 25 + size,
                "L", 18 + x_offset, 25,
                "L", x_offset, 25 - size,
                "z"]

    @staticmethod
    def polygon_port_out(size: float, x_offset: float = 0):
        return ["M", x_offset, 25 + size,
                "L", 10 + x_offset, 25 + size*.85,
                "L", 18 + x_offset, 25,
                "L", 10 + x_offset, 25-size*.85,
                "L", x_offset, 25-size,
                "z"]

    @staticmethod
    def bs_symbolic_mpath(compact: bool):
        coeff = 1 if compact else 2
        return ["M", 6.4721*coeff, 25,
                "c", 6.8548*coeff, 0, 6.8241*coeff, 25, 13.68*coeff, 25,
                "m", 0.0009*coeff, 0,
                "c", -6.8558*coeff, 0, -6.825*coeff, 25, -13.68*coeff, 25,
                "m", 13.68*coeff, -25,
                "h", 10.9423*coeff,
                "m", 0, 0,
                "c", 6.8558*coeff, 0, 6.825*coeff, -25, 13.68*coeff, -25,
                "m", -13.6799*coeff, 25,
                "c", 6.8558*coeff, 0, 6.825*coeff, 25, 13.68*coeff, 25,
                "m", -44.7741*coeff, -50,
                "h", 6.5*coeff,
                "m", 0.0009*coeff, 50,
                "h", -6.5009*coeff,
                "m", 43.8227*coeff, 0,
                "h", 6.1773*coeff,
                "m", -6.4028*coeff, -50,
                "h", 6.4028*coeff]

    @staticmethod
    def rounded_corner_square(radius: float, length_ratio: float):
        return ["M", 0, radius,
                "c", 0, 0, 0, -radius, radius, -radius,
                "l", length_ratio * radius, 0,
                "c", radius, 0, radius, radius, radius, radius,
                "l", 0, length_ratio * radius,
                "c", 0, 0, 0, radius, -radius, radius,
                "l", -length_ratio * radius, 0,
                "c", -radius, 0, -radius, -radius, -radius, -radius,
                "l", 0, -length_ratio * radius,
                "z"]
