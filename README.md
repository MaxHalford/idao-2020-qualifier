## To do

- [ ] Metric learning as a preprocessor, in addition to standard scaling
- [ ] Naive Bayes for leaf predictions
- [ ] [Tree ensemble features](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py)
- [x] Detect satellites that have shifts
- [x] Calculate cycle order
- [ ] Use monotonically increasing weights in polyfit

- Predict yp / yt
- Predict (yt - yp) / yp
- Predict (yt+1 - yp+1) / (yt - yp)


Interesting:

- Full sequence: 250
- Two big bubbles: 249

## Observations

- Most time series seem to have a periodicity of 24
- There are duplicates: some observations are extremely close together
- There are also some missing simulation values, for instance for satellite 4, at the 9th cycle, index 19, a shift begins

## Vocabulary

- COE (Classical Orbital Element): state vector expressed as (a, e, i, ω, Ω, f)
- ECI (Earth-Centered Inertial): state vector expressed as (X, Y, Z, VX, VY, VZ)
- WGS (World Geodetic System): coordinate system based around planet Earth.

## Resources

- [Wiki article on orbital state vectors](https://www.wikiwand.com/en/Orbital_state_vectors)
- [Wiki article on the World Geodetic System](https://www.wikiwand.com/en/World_Geodetic_System)
- [Improving Orbit Prediction Accuracy through Supervised Machine Learning](https://arxiv.org/pdf/1801.04856.pdf)
- [Orbital Mechanics](http://www.braeunig.us/space/orbmech.htm)
