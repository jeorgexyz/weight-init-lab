# Weight Init Lab

Interactive, in-browser demos of neural network weight initialization. No build step, no dependencies.

- **Signal propagation:** push random inputs through a deep, untrained network and watch the forward signal and backward gradient grow, shrink or hold steady under zeros, tiny/large normal, LeCun, Xavier and He initialization, with tanh, ReLU, sigmoid or linear activations.
- **Derivation:** the short variance argument behind Xavier and He.
- **Field guide:** a table of schemes, what they are built for, and the actual weight std at the selected width.
- **Training race:** four identical networks, differing only in initialization, trained side by side on two spirals, rings or XOR.

Everything is seeded, so results are repeatable.

## Run

Open `index.html` in a browser, or serve the folder:

```
python -m http.server
```

## References

- LeCun, Bottou, Orr, Müller. *Efficient BackProp* (1998)
- Glorot, Bengio. *Understanding the difficulty of training deep feedforward neural networks* (AISTATS 2010)
- He, Zhang, Ren, Sun. *Delving Deep into Rectifiers* (ICCV 2015)
