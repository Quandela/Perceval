import gc
import time
import perceval as pcvl
import psutil

from perceval.backends import BACKEND_LIST
from perceval.backends._slos_v2 import SLOSV2Backend
from perceval.backends._slos_v3 import SLOSV3Backend
from perceval.utils.dist_metrics import tvd_dist
from perceval.utils.postselect import PostSelect
import exqalibur as xq

from perceval.utils import get_logger
get_logger().set_level(pcvl.logging.level.warn, pcvl.logging.channel.general)
# get_logger().set_level(pcvl.logging.level.info, pcvl.logging.channel.general)

# BACKEND_LIST[ "SLOS_V2_PS" ] = SLOSV2Backend
# BACKEND_LIST[ "SLOS_V3_PS" ] = SLOSV3Backend

def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def do_compute(backend, method):
    if method == 1:
        return backend.all_prob_ampli()
    if method == 2:
        return backend.all_prob()
    if method == 3:
        return backend.prob_distribution()

def set_mask(backend, mask_str, m, n):
    backend.set_mask((mask_str + "*" * m)[:m], n)
    # if backend.name == "SLOS_V2_PS" or backend.name == "SLOS_V3_PS":
    #     backend.set_post_select(ps)

def bench(m, n, backends, mask_str, method):
    u1 = pcvl.Matrix.random_unitary(m)
    photons = [ 0 ] * m
    photons[0] = n
    bs = pcvl.BasicState(photons)
    photons[0] = n - 1
    photons[1] = 1
    bs2 = pcvl.BasicState(photons)
    circuit = pcvl.Circuit(m) // pcvl.components.Unitary(u1)

    tvds = {}
    result = []
    refbsd = None
    ref = "-"
    for backend_name in backends:
        # get_logger().warn(backend_name)
        if backend_name == "SLAP" and n > 15:
            print("\t-- ", end='', flush=True)
            continue
        if backend_name == "SLOS" and n > 18:
            print("\t-- ", end='', flush=True)
            continue
        start = time.time()
        backend = pcvl.backends.BackendFactory.get_backend(backend_name)
        set_mask(backend, mask_str, m, n)
        backend.set_circuit(circuit)
        backend.set_input_state(bs)
        bsd = do_compute(backend, method)
        end = time.time()

        backend.set_input_state(bs2)
        bsd = do_compute(backend, method)
        end2 = time.time()

        # if backend_name == "SLOS_V2" or backend_name == "SLOS_V2_PS":
        #     print(bsd)
        # if refbsd:
        #     tvds[backend_name] = tvd_dist(bsd, refbsd)
        #     pass
        # else:
        #     refbsd = bsd
        #     ref = backend_name
        print(f"\t{end-start:.4f} {end2-end:.4f}", end='', flush=True)
        result.append(end - start)
    for k, v in tvds.items():
        print(f"\n{k} TVD against {ref}: {v}", end='')

    return result


def bench_mem(m, n, backend_name, mask_str, method):
    u1 = pcvl.Matrix.random_unitary(m)
    photons = [ 0 ] * m
    photons[0] = n
    bs = pcvl.BasicState(photons)
    circuit = pcvl.Circuit(m) // pcvl.components.Unitary(u1)

    # if backend_name == "SLOS" and n > 12:
    #     break
    gc.collect()
    mem_1 = psutil.Process().memory_info().rss
    backend = pcvl.backends.BackendFactory.get_backend(backend_name)
    set_mask(backend, mask_str, m, n)
    backend.set_circuit(circuit)
    backend.set_input_state(bs)
    bsd = do_compute(backend, method)
    mem_2 = psutil.Process().memory_info().rss
    backend = None
    bsd = None
    gc.collect()
    print(f"{backend_name:12}\t{human_readable_size(mem_2-mem_1)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmarks of strong simulation backends')
    parser.add_argument('--modes', '-m',
                        type=int, required=True, action='store',
                        help='number of modes')
    parser.add_argument('--nphotons', '-n',
                        type=int, action='store', default=20,
                        help='number of modes')
    parser.add_argument('--minphotons', '-i',
                        type=int, action='store', default=4,
                        help='number of modes')
    parser.add_argument('--memusage', '-u', action='store_true', default=False)
    parser.add_argument('--backends', '-b',
                        type=str, action='store', default='SLOS SLAP SLOS_CPP SLOS_V2 SLOS_V3',
                        help='backend used')
    parser.add_argument('--mask', '-k',
                        type=str, action='store', default='',
                        help='mask')
    parser.add_argument('--compute', '-c',
                        type=int, action='store', default='3',
                        help='Computations: 0 -> all_prob_ampli / 1 -> all_prob / 2 -> prob_distribution')
    args = parser.parse_args()

    backend_list = args.backends.split(" ")
    if args.memusage:
        bench_mem(args.modes, args.nphotons, backend_list[0], args.mask, args.compute)
    else:
        print('\t', end = '', flush=False)
        for b in backend_list:
            print(f"{b:<16}", end = '', flush=False)
        print('', flush=True)
        for n in range(args.minphotons, args.nphotons + 1):
            print(f"{n}", end='')
            bench(args.modes, n, backend_list, args.mask, args.compute)
            print("")
