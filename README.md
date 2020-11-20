# IDAO 2020 qualification phase

This repository contains my team's solution to the [2020 edition](https://idao.world/results/) of the [International Data Analysis Olympiad (IDAO)](https://idao.world/). Our team is called Data O Plomo. [Here](https://official.contest.yandex.ru/contest/16669/problems/)'s some more information on the contest.

Overall we ranked 2nd on track 1, 1st on track 2, and 1st overall. We used the same model for both tracks. Our model is very simple, and is basically an [autoregressive](https://www.wikiwand.com/en/Autoregressive_model) linear regression with a few bells and whistles. You may find more information in [these slides](https://maxhalford.github.io/slides/idao-2020-qualifiers.pdf) which we presented during an online webinar to other contestants.

## Usage

You first want to download the competition data from [here](https://yadi.sk/d/0zYx00gSraxZ3w). Then, unzip the data folder into `data/`. You should thus have `data/train.csv`, `data/Track 1`, and `data/Track 2` on your path.

We built several simple models which are each contained in a Jupyter notebook. You can either open them and execute them manually, or programmatically by using `nbconvert`.

```sh
jupyter nbconvert \
    --execute auto-regression.ipynb \
    --to notebook \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --debug
    
jupyter nbconvert \
    --execute cycle_regression.ipynb \
    --to notebook \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --debug
```

Each notebook will produces validation scores as well as submission files, both of which are stored in the `results` directory. For instance, `auto-regression.ipynb` will output `results/ar_track_1.csv` (which is the submission file) and `results/ar_val_scores.csv` (which are the validation scores).

We can now blend the submissions. This will produce a submission file named `track_1_blended.csv` in the `results` directory.

```sh
python results/blend_track_1.py
```

Finally, the submission for track 2 can be obtained by zipping the `track_2` directory. The latter contains a file named `ar_models.pkl` which is produced by the `auto-regression.ipynb` notebook.

```sh
rm -f track_2/*.csv track_2/*.dat  # remove unnecessary artifacts
zip -jr results/track_2.zip track_2
```

## Track 2 performance profiling

The goal of track 2 was to implement a model which could make predictions on the test set in less than 60 seconds with under 500 MB of RAM. We used the [`memory_profiler`](https://github.com/pythonprofilers/memory_profiler) package for measuring the memory consumption of our script for track 2.

```sh
cd track_2
pip install memory_profiler
mprof run python main.py
mprof plot --output ./results/track_2_memory_usage.png
```

![track_2_memory_usage](results/track_2_memory_usage.png)

As for speed, we used a rule of thumb, which is that the Yandex machine used for running our code is 20 seconds slower than our machine. We thus checked that our code took at most 40 seconds to run on our machine. For reference, our CPU model is `Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz`.
