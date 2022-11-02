window.BENCHMARK_DATA = {
  "lastUpdate": 1667380967620,
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
      },
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
          "id": "4ffcc3bc0a77a21cd624fd7660052a7fcbec53f4",
          "message": "Merge branch 'releaseUP0.7' of github.com:julienCALISTO/Perceval into releaseUP0.7",
          "timestamp": "2022-11-02T09:13:01Z",
          "url": "https://github.com/julienCALISTO/Perceval/commit/4ffcc3bc0a77a21cd624fd7660052a7fcbec53f4"
        },
        "date": 1667380957137,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_clifford_6",
            "value": 74.76245870849317,
            "unit": "iter/sec",
            "range": "stddev: 0.00007493726823952008",
            "extra": "mean: 13.375697071428684 msec\nrounds: 70"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_slos_6",
            "value": 9.687817695834923,
            "unit": "iter/sec",
            "range": "stddev: 0.004308326686454461",
            "extra": "mean: 103.22242133333386 msec\nrounds: 9"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_naive_6",
            "value": 1.458387677962571,
            "unit": "iter/sec",
            "range": "stddev: 0.04296875116554739",
            "extra": "mean: 685.6887336 msec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_clifford_8",
            "value": 191.85394997994328,
            "unit": "iter/sec",
            "range": "stddev: 0.000022592700490091415",
            "extra": "mean: 5.212298209677422 msec\nrounds: 186"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_slos_8",
            "value": 2.7041327270201947,
            "unit": "iter/sec",
            "range": "stddev: 0.04874665488697985",
            "extra": "mean: 369.80433319999975 msec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_naive_8",
            "value": 0.3087351270808198,
            "unit": "iter/sec",
            "range": "stddev: 0.41776394264693045",
            "extra": "mean: 3.2390224249999995 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_mplot_6",
            "value": 0.35175547909338345,
            "unit": "iter/sec",
            "range": "stddev: 0.022596269375095134",
            "extra": "mean: 2.842883933400002 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_mplot_12",
            "value": 0.08114719586813436,
            "unit": "iter/sec",
            "range": "stddev: 0.038946531497941535",
            "extra": "mean: 12.323284733400005 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_svg_6",
            "value": 0.3740940704088646,
            "unit": "iter/sec",
            "range": "stddev: 0.0035902705041854542",
            "extra": "mean: 2.673124433400011 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_svg_12",
            "value": 0.08476632698687349,
            "unit": "iter/sec",
            "range": "stddev: 0.08970218382583614",
            "extra": "mean: 11.797137325 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper",
            "value": 77.65577973359404,
            "unit": "iter/sec",
            "range": "stddev: 0.00010687097992226273",
            "extra": "mean: 12.877341563378804 msec\nrounds: 71"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper_comp_naive",
            "value": 586.1155204005839,
            "unit": "iter/sec",
            "range": "stddev: 0.000029650329607027763",
            "extra": "mean: 1.7061483021581554 msec\nrounds: 556"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper_comp_direct",
            "value": 173.49647491020613,
            "unit": "iter/sec",
            "range": "stddev: 0.000023082155337772796",
            "extra": "mean: 5.763805867050351 msec\nrounds: 173"
          }
        ]
      }
    ]
  }
}