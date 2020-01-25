# IDAO 2020 qualification phase

## Usage

```sh
# Submission for track 1 and models for track 2
jupyter nbconvert --to notebook --inplace --execute auto-regression.ipynb

# Submission for track 2
rm -f track_2/*.csv track_2/*.dat
zip -jr results/track_2.zip track_2
```

## Memory usage

We used the `memory_profiler` package to measure the memory consumption of our script for track 2.

```sh
cd track_2
pip install memory_profiler
mprof run python main.py
mprof plot --output ../results/track_2_memory_usage.png
```

![track_2_memory_usage](results/track_2_memory_usage.png)

## To do

- [ ] Metric learning as a preprocessor, in addition to standard scaling
- [ ] Naive Bayes for leaf predictions
- [ ] [Tree ensemble features](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py)
- [x] Detect satellites that have shifts
- [x] Calculate cycle order
- [ ] Normalize coordinates (check out DIPY)
- [ ] Use monotonically increasing weights
- [ ] Make polynomial features work
- [ ] Predict all the targets simultaneously instead of separately (still per satellite though)

## Vocabulary

- COE (Classical Orbital Element): state vector expressed as (a, e, i, ω, Ω, f)
- ECI (Earth-Centered Inertial): state vector expressed as (X, Y, Z, VX, VY, VZ)
- WGS (World Geodetic System): coordinate system based around planet Earth.

## Resources

- [Wiki article on orbital state vectors](https://www.wikiwand.com/en/Orbital_state_vectors)
- [Wiki article on the World Geodetic System](https://www.wikiwand.com/en/World_Geodetic_System)
- [Improving Orbit Prediction Accuracy through Supervised Machine Learning](https://arxiv.org/pdf/1801.04856.pdf)
- [Orbital Mechanics](http://www.braeunig.us/space/orbmech.htm)
