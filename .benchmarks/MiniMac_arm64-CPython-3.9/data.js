window.BENCHMARK_DATA = {
  "lastUpdate": 1667379959608,
  "repoUrl": "https://github.com/julienCALISTO/Perceval",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "name": "Julien Calisto",
            "username": "julienCALISTO",
            "email": "julien.calisto@quandela.com"
          },
          "committer": {
            "name": "Julien Calisto",
            "username": "julienCALISTO",
            "email": "julien.calisto@quandela.com"
          },
          "id": "a6a820ac74ce1b070ce884dd5c8bd1c4ef3a8800",
          "message": "download artefact in .benchmarks/log",
          "timestamp": "2022-11-02T08:55:45Z",
          "url": "https://github.com/julienCALISTO/Perceval/commit/a6a820ac74ce1b070ce884dd5c8bd1c4ef3a8800"
        },
        "date": 1667379956804,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_clifford_6",
            "value": 74.68246270456487,
            "unit": "iter/sec",
            "range": "stddev: 0.00007563292083602356",
            "extra": "mean: 13.39002442857145 msec\nrounds: 70"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_slos_6",
            "value": 9.722683136819297,
            "unit": "iter/sec",
            "range": "stddev: 0.007239928868264735",
            "extra": "mean: 102.85226680000008 msec\nrounds: 10"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_naive_6",
            "value": 1.4743054362801509,
            "unit": "iter/sec",
            "range": "stddev: 0.060179286015943106",
            "extra": "mean: 678.2854999999997 msec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_clifford_8",
            "value": 192.07516935501943,
            "unit": "iter/sec",
            "range": "stddev: 0.0000621206943141936",
            "extra": "mean: 5.206295032085403 msec\nrounds: 187"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_slos_8",
            "value": 2.9833037164066867,
            "unit": "iter/sec",
            "range": "stddev: 0.053898318974957796",
            "extra": "mean: 335.1988584000004 msec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_naive_8",
            "value": 0.3199439125318813,
            "unit": "iter/sec",
            "range": "stddev: 0.5767679579286196",
            "extra": "mean: 3.125547825199998 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_mplot_6",
            "value": 0.35547948297147836,
            "unit": "iter/sec",
            "range": "stddev: 0.02219785617579117",
            "extra": "mean: 2.8131018747999987 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_mplot_12",
            "value": 0.08206981810235411,
            "unit": "iter/sec",
            "range": "stddev: 0.04468203084387715",
            "extra": "mean: 12.184747366599996 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_svg_6",
            "value": 0.3790810479799976,
            "unit": "iter/sec",
            "range": "stddev: 0.0011665625418168612",
            "extra": "mean: 2.6379583082000067 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_svg_12",
            "value": 0.0857667628467078,
            "unit": "iter/sec",
            "range": "stddev: 0.08519867087271171",
            "extra": "mean: 11.659528316200005 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper",
            "value": 77.05179731921564,
            "unit": "iter/sec",
            "range": "stddev: 0.0001288687898256509",
            "extra": "mean: 12.978282594202561 msec\nrounds: 69"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper_comp_naive",
            "value": 583.7205523138581,
            "unit": "iter/sec",
            "range": "stddev: 0.0000288591762539695",
            "extra": "mean: 1.7131485195030696 msec\nrounds: 564"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper_comp_direct",
            "value": 172.96612407982346,
            "unit": "iter/sec",
            "range": "stddev: 0.00002426875243983962",
            "extra": "mean: 5.781478918603173 msec\nrounds: 172"
          }
        ]
      }
    ]
  }
}