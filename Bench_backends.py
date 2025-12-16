import gc
import time
import perceval as pcvl
import psutil
import exqalibur

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
    bs = pcvl.BasicState(photons)
    circuit = pcvl.Circuit(m) // pcvl.components.Unitary(u1)

    result = []
    for backend_name in backends:
        if backend_name == "SLAP" and n > 15:
            print("\t-- ", end='', flush=True)
            continue
        if backend_name == "SLOS" and n > 18:
            print("\t-- ", end='', flush=True)
            continue
        backend = pcvl.backends.BackendFactory.get_backend(backend_name)
        backend.set_circuit(circuit)
        backend.set_input_state(bs)
        start = time.time()
        bsd = backend.prob_distribution()
        end = time.time()
        print(f"\t{end-start}", end='', flush=True)
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
        backend = pcvl.backends.BackendFactory.get_backend(backend_name)
        backend.set_circuit(circuit)
        backend.set_input_state(bs)
        gc.collect()
        mem_1 = psutil.Process().memory_info().rss
        bsd = backend.prob_distribution()
        mem_2 = psutil.Process().memory_info().rss
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
    args = parser.parse_args()
    backends = [ "SLOS", "SLOS_CPP", "SLAP", "SLOS_V2" ]

    if args.memusage:
        bench_mem(args.modes, args.nphotons, backends)
    else:
        print('\t', "\t".join(backends))
        for n in range(args.minphotons, args.nphotons + 1):
            print(f"{n}", end='')
            bench(args.modes, n, backends)
            print("")
