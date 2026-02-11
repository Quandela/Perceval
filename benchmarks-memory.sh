SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR
python ${SCRIPT_DIR}/Bench_backends.py -m 20 -n 10 -u -b SLOS
python ${SCRIPT_DIR}/Bench_backends.py -m 20 -n 10 -u -b SLAP
python ${SCRIPT_DIR}/Bench_backends.py -m 20 -n 10 -u -b SLOS_CPP
python ${SCRIPT_DIR}/Bench_backends.py -m 20 -n 10 -u -b SLOS_V2
# python ${SCRIPT_DIR}/Bench_backends.py -m 20 -n 10 -u -b SLOS_V2_PS
python ${SCRIPT_DIR}/Bench_backends.py -m 20 -n 10 -u -b SLOS_V3
# python ${SCRIPT_DIR}/Bench_backends.py -m 20 -n 10 -u -b SLOS_V3_PS
