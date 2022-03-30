import perceval as pcvl

u = pcvl.Matrix.random_unitary(10)
sim = pcvl.BackendFactory().get_backend("SLOS")(pcvl.Circuit(10, u), n=3, mask=["3         "])

print(sim.prob(input_state=pcvl.BasicState([1,0,1,0,0,1,0,0,0,0]),
               output_state=pcvl.BasicState([3,0,0,0,0,0,0,0,0,0])))
print(sim.prob(input_state=pcvl.BasicState([1,0,1,0,0,0,0,0,0,1]),
               output_state=pcvl.BasicState([3,0,0,0,0,0,0,0,0,0])))