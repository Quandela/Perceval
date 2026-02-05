import gc
import time
import perceval as pcvl
import psutil
from perceval.backends import BACKEND_LIST
from perceval.backends._slos_v2 import SLOSV2Backend
from perceval.backends._slos_v3 import SLOSV3Backend
from perceval.utils.postselect import PostSelect
import exqalibur as xq
BACKEND_LIST[ "SLOS_V2_PS" ] = SLOSV2Backend
BACKEND_LIST[ "SLOS_V3_PS" ] = SLOSV3Backend

def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def bench(m, n, backends):
    u1 = pcvl.Matrix.random_unitary(m)
    photons = [ 0 ] * m
    photons[0] = n
    ps = PostSelect('[0]==1 & [1]==1 & [2]==1 & [3]==1')
    # ps = PostSelect('[0]==0 & [1]==0 & [2]==0 & [3]==0')
    # ps = PostSelect('[0]==1 & [1]==1 & [2]==0 & [3]==0')
    mask_str = "1111" + "*" * (m-4)
    # mask_str = "0000" + "*" * (m-4)
    # mask_str = "1100" + "*" * (m-4)
    mask = xq.FSMask(m, n, [mask_str])
    bs = pcvl.BasicState(photons)
    photons[0] = n - 1
    photons[1] = 1
    bs2 = pcvl.BasicState(photons)
    circuit = pcvl.Circuit(m) // pcvl.components.Unitary(u1)

    result = []
    for backend_name in backends:
        if backend_name == "SLAP" and n > 15:
            print("\t-- ", end='', flush=True)
            continue
        if backend_name == "SLOS" and n > 18:
            print("\t-- ", end='', flush=True)
            continue
        start = time.time()
        backend = pcvl.backends.BackendFactory.get_backend(backend_name)
        if backend_name == "SLOS_V2_PS" or backend_name == "SLOS_V3_PS":
            backend.set_post_select(ps)
            pass
        else:
            backend.set_mask(mask_str, n)
            pass
        backend.set_circuit(circuit)
        backend.set_input_state(bs)
        bsd = backend.all_prob()
        end = time.time()

        backend.set_input_state(bs2)
        bsd = backend.all_prob()
        # bsd = backend.prob_distribution()
        end2 = time.time()
        print(f"\t{end-start:.4f} {end2-end:.4f}", end='', flush=True)
        result.append(end - start)

    return result


def bench_mem(m, n, backends):
    u1 = pcvl.Matrix.random_unitary(m)
    photons = [ 0 ] * m
    photons[0] = n
    bs = pcvl.BasicState(photons)
    circuit = pcvl.Circuit(m) // pcvl.components.Unitary(u1)

    for backend_name in backends:
        # if backend_name == "SLOS" and n > 12:
        #     break
        gc.collect()
        mem_1 = psutil.Process().memory_info().rss
        backend = pcvl.backends.BackendFactory.get_backend(backend_name)
        backend.set_circuit(circuit)
        backend.set_input_state(bs)
        bsd = backend.prob_distribution()
        mem_2 = psutil.Process().memory_info().rss
        backend = None
        bsd = None
        gc.collect()
        print(f"{backend_name}\t{mem_1}\t{mem_2}\t{human_readable_size(mem_2-mem_1)}")

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
    parser.add_argument('--backend', '-b',
                        type=str, action='store', default='SLOS',
                        help='backend used')
    args = parser.parse_args()
    backends = [
            "SLOS",
            "SLAP",
            "SLOS_CPP",
            "SLOS_V2",
            "SLOS_V2_PS",
            "SLOS_V3",
            "SLOS_V3_PS",
        ]

    if args.memusage:
        bench_mem(args.modes, args.nphotons, [args.backend])
    else:
        print('\t', "\t".join(backends))
        for n in range(args.minphotons, args.nphotons + 1):
            print(f"{n}", end='')
            bench(args.modes, n, backends)
            print("")
