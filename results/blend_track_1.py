import collections
import os
import matplotlib.pyplot as plt

import pandas as pd
import tqdm


SUBMISSIONS = ['ar', 'cr', 'cr_speed']

SAT_VAR_MODELS = collections.defaultdict(lambda: collections.defaultdict(lambda: 'ar'))


if __name__ == '__main__':

    here = os.path.dirname(os.path.realpath(__file__))

    # The predictions will default to those of the auto-regressive model, we can thus manually
    # specify overrides here
    #SAT_VAR_MODELS[0]['x'] = 'ar'
    model = SUBMISSIONS[0]
    val_scores = pd.read_csv(os.path.join(here, f'{model}_val_scores.csv'), names=["sat_id", "variable", f"smape_{model}"], header=0)
    for model in SUBMISSIONS[1:]:
        val_scores = val_scores.merge(pd.read_csv(os.path.join(here, f'{model}_val_scores.csv'), names=["sat_id", "variable", f"smape_{model}"], header=0), on=["sat_id", "variable"])
    
    val_scores["best_model"] = val_scores[[f"smape_{m}" for m in SUBMISSIONS]].idxmin(axis=1).apply(lambda s: s.replace('smape_', ''))

#     val_scores.best_model.hist()
#     plt.show()

    for sat in tqdm.tqdm(val_scores.sat_id):
        for var in ('x', 'y', 'z', 'Vx', 'Vy', 'Vz'):
            SAT_VAR_MODELS[sat][var] = val_scores.query('sat_id == @sat and variable == @var')['best_model'].values[0]
    
    print(SAT_VAR_MODELS[0]['x'])
    

    # We use the test file to map satellites IDs to row IDs
    sat_to_ids = (
        pd.read_csv(os.path.join(here, '../data/Track 1/test.csv'), usecols=['sat_id', 'id'])
        .groupby('sat_id')['id']
        .apply(set)
        .to_dict()
    )

    # Load the track 1 submission files of each method
    subs = {
        model: pd.read_csv(os.path.join(here, f'{model}_track_1.csv')).set_index('id')
        for model in SUBMISSIONS
    }

    blend = subs[SUBMISSIONS[0]]

    for sat_id in tqdm.tqdm(sat_to_ids.keys()):
        for var in ('x', 'y', 'z', 'Vx', 'Vy', 'Vz'):

            ids = sat_to_ids[sat_id]
            model = SAT_VAR_MODELS[sat_id][var]
            blend.loc[ids, var] = subs[model].loc[ids, var].values

    blend.to_csv(os.path.join(here, 'track_1_blended_ar_cr_speedCR.csv'))
