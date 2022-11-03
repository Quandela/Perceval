window.BENCHMARK_DATA = {
  "lastUpdate": 1667489965320,
  "repoUrl": "https://github.com/julienCALISTO/Perceval",
  "entries": {
    "Automated report": [
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
          "id": "e416f952d4373576b1611a582b0785f73adcc176",
          "message": "modiff name for confignotlinux",
          "timestamp": "2022-11-02T10:17:56Z",
          "url": "https://github.com/julienCALISTO/Perceval/commit/e416f952d4373576b1611a582b0785f73adcc176"
        },
        "date": 1667384755157,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_clifford_6",
            "value": 74.37625802071773,
            "unit": "iter/sec",
            "range": "stddev: 0.00006772539764722832",
            "extra": "mean: 13.445150732394296 msec\nrounds: 71"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_slos_6",
            "value": 9.540127694170357,
            "unit": "iter/sec",
            "range": "stddev: 0.006938544367898037",
            "extra": "mean: 104.82039989999983 msec\nrounds: 10"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_naive_6",
            "value": 1.4495485875220246,
            "unit": "iter/sec",
            "range": "stddev: 0.04034673208887318",
            "extra": "mean: 689.8699419999992 msec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_clifford_8",
            "value": 191.8117893979662,
            "unit": "iter/sec",
            "range": "stddev: 0.000024987049350762703",
            "extra": "mean: 5.213443882352953 msec\nrounds: 187"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_slos_8",
            "value": 3.3461152146102022,
            "unit": "iter/sec",
            "range": "stddev: 0.05292667417137648",
            "extra": "mean: 298.85402500000066 msec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_naive_8",
            "value": 0.28087064207255114,
            "unit": "iter/sec",
            "range": "stddev: 0.455152733000682",
            "extra": "mean: 3.560357866599999 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_mplot_6",
            "value": 0.35486608702366806,
            "unit": "iter/sec",
            "range": "stddev: 0.02148648304676019",
            "extra": "mean: 2.8179644000000037 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_mplot_12",
            "value": 0.08182678438049841,
            "unit": "iter/sec",
            "range": "stddev: 0.03510441855415562",
            "extra": "mean: 12.220937283199996 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_svg_6",
            "value": 0.37679877018331454,
            "unit": "iter/sec",
            "range": "stddev: 0.0027958224209043364",
            "extra": "mean: 2.6539364751999983 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_svg_12",
            "value": 0.08546594043041289,
            "unit": "iter/sec",
            "range": "stddev: 0.08452783048148668",
            "extra": "mean: 11.700567441999993 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper",
            "value": 77.13010800467379,
            "unit": "iter/sec",
            "range": "stddev: 0.00013917882984106937",
            "extra": "mean: 12.965105661973192 msec\nrounds: 71"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper_comp_naive",
            "value": 586.5751684960582,
            "unit": "iter/sec",
            "range": "stddev: 0.0000298301086858844",
            "extra": "mean: 1.7048113416801074 msec\nrounds: 559"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper_comp_direct",
            "value": 174.73024498273938,
            "unit": "iter/sec",
            "range": "stddev: 0.000028919978829168122",
            "extra": "mean: 5.723107640001217 msec\nrounds: 175"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-action-benchmark",
            "username": "github",
            "email": "github@users.noreply.github.com"
          },
          "committer": {
            "name": "github-action-benchmark",
            "username": "github",
            "email": "github@users.noreply.github.com"
          },
          "id": "00c65785b1275f49f6748b70abe2a69e8ddd3dde",
          "message": "Automated benchmark log report",
          "timestamp": "2022-11-02T10:26:11Z",
          "url": "https://github.com/julienCALISTO/Perceval/commit/00c65785b1275f49f6748b70abe2a69e8ddd3dde"
        },
        "date": 1667385092232,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_clifford_6",
            "value": 74.55802707749503,
            "unit": "iter/sec",
            "range": "stddev: 0.00010439228304562028",
            "extra": "mean: 13.412372070422517 msec\nrounds: 71"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_slos_6",
            "value": 9.053031900417936,
            "unit": "iter/sec",
            "range": "stddev: 0.005441399990968064",
            "extra": "mean: 110.46023155555594 msec\nrounds: 9"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_naive_6",
            "value": 1.4474600933855468,
            "unit": "iter/sec",
            "range": "stddev: 0.029520343997476727",
            "extra": "mean: 690.8653333999994 msec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_clifford_8",
            "value": 193.58620252096247,
            "unit": "iter/sec",
            "range": "stddev: 0.000022843758022634565",
            "extra": "mean: 5.165657402116326 msec\nrounds: 189"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_slos_8",
            "value": 3.3442813958629336,
            "unit": "iter/sec",
            "range": "stddev: 0.030805933905870865",
            "extra": "mean: 299.01789999999903 msec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_naive_8",
            "value": 0.3062384258400425,
            "unit": "iter/sec",
            "range": "stddev: 0.34199734548037725",
            "extra": "mean: 3.265429533399998 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_mplot_6",
            "value": 0.3552720105701761,
            "unit": "iter/sec",
            "range": "stddev: 0.022068364013196323",
            "extra": "mean: 2.8147446751999965 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_mplot_12",
            "value": 0.08196956246185277,
            "unit": "iter/sec",
            "range": "stddev: 0.04017464502551873",
            "extra": "mean: 12.199650333199994 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_svg_6",
            "value": 0.3782918197380833,
            "unit": "iter/sec",
            "range": "stddev: 0.000756348070351251",
            "extra": "mean: 2.643461866799993 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_svg_12",
            "value": 0.08576097355850604,
            "unit": "iter/sec",
            "range": "stddev: 0.08043572444462681",
            "extra": "mean: 11.660315391799992 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper",
            "value": 77.44247616431157,
            "unit": "iter/sec",
            "range": "stddev: 0.00010145551052933792",
            "extra": "mean: 12.91281025000125 msec\nrounds: 72"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper_comp_naive",
            "value": 587.5041298178993,
            "unit": "iter/sec",
            "range": "stddev: 0.000029913868087748684",
            "extra": "mean: 1.702115694590873 msec\nrounds: 573"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper_comp_direct",
            "value": 175.1510586801507,
            "unit": "iter/sec",
            "range": "stddev: 0.00001727444901138054",
            "extra": "mean: 5.709357440003455 msec\nrounds: 175"
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
          "id": "325b1686aed73047bfa0a2fb95dc38453f3d87c1",
          "message": "modiff path name",
          "timestamp": "2022-11-03T15:31:21Z",
          "url": "https://github.com/julienCALISTO/Perceval/commit/325b1686aed73047bfa0a2fb95dc38453f3d87c1"
        },
        "date": 1667489950661,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_clifford_6",
            "value": 74.91026291228467,
            "unit": "iter/sec",
            "range": "stddev: 0.00008493707198561754",
            "extra": "mean: 13.349305704225584 msec\nrounds: 71"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_slos_6",
            "value": 9.406537209114763,
            "unit": "iter/sec",
            "range": "stddev: 0.006022529265443195",
            "extra": "mean: 106.30904633333277 msec\nrounds: 9"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_naive_6",
            "value": 1.4343420427418856,
            "unit": "iter/sec",
            "range": "stddev: 0.03482910819537976",
            "extra": "mean: 697.1837750000007 msec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_clifford_8",
            "value": 192.10897055283917,
            "unit": "iter/sec",
            "range": "stddev: 0.00002349519004915938",
            "extra": "mean: 5.205378994652164 msec\nrounds: 187"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_slos_8",
            "value": 3.036762900441965,
            "unit": "iter/sec",
            "range": "stddev: 0.03402440292790931",
            "extra": "mean: 329.29801659999924 msec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_bosonsampling.py::test_bosonsampling_naive_8",
            "value": 0.32642489052944373,
            "unit": "iter/sec",
            "range": "stddev: 0.3366779596459719",
            "extra": "mean: 3.063491875199999 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_mplot_6",
            "value": 0.36268687795641114,
            "unit": "iter/sec",
            "range": "stddev: 0.021334510226804833",
            "extra": "mean: 2.7571992834 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_mplot_12",
            "value": 0.08370592561679693,
            "unit": "iter/sec",
            "range": "stddev: 0.03117478831908667",
            "extra": "mean: 11.946585532999995 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_svg_6",
            "value": 0.38672051385582873,
            "unit": "iter/sec",
            "range": "stddev: 0.0005079217814646373",
            "extra": "mean: 2.585846791600005 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_pdisplay.py::test_pdisplay_svg_12",
            "value": 0.08765195808798382,
            "unit": "iter/sec",
            "range": "stddev: 0.08737466838772094",
            "extra": "mean: 11.408758250399995 sec\nrounds: 5"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper",
            "value": 77.6766797211056,
            "unit": "iter/sec",
            "range": "stddev: 0.00009241360130513037",
            "extra": "mean: 12.873876736112463 msec\nrounds: 72"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper_comp_naive",
            "value": 585.5962453816869,
            "unit": "iter/sec",
            "range": "stddev: 0.00003027024068482606",
            "extra": "mean: 1.7076612220220233 msec\nrounds: 554"
          },
          {
            "name": "benchmark/benchmark_stepper.py::test_stepper_comp_direct",
            "value": 173.8243157640483,
            "unit": "iter/sec",
            "range": "stddev: 0.000029545127403028322",
            "extra": "mean: 5.752935057471561 msec\nrounds: 174"
          }
        ]
      }
    ]
  }
}