import os
in_notebook = False
in_pycharm_or_spyder = "PYCHARM_HOSTED" in os.environ or 'SPY_PYTHONPATH' in os.environ

try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        in_notebook = True
        from IPython.display import HTML, display
except ImportError:
    pass
except AttributeError:
    pass


def pdisplay(o, output_format=None, **opts):
    if output_format is None:
        if in_notebook:
            output_format = "html"
        elif hasattr(o, "delay_circuit") and in_pycharm_or_spyder:
            # characterize ACCircuit objects
            output_format = "mplot"
        else:
            output_format = "text"

    r = o.pdisplay(output_format=output_format, **opts)
    if in_notebook and output_format != "text":
        display(HTML(r))
    else:
        print(r)
