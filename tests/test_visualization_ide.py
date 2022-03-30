import os

os.environ["PYCHARM_HOSTED"] = "1"
import perceval as pcvl
import perceval.lib.phys as phys

def test_ide_visualization(capfd):
    pcvl.pdisplay(phys.PS(0).definition())
    out, err = capfd.readouterr()
    print(out)
    assert out.find("matrix") == -1
