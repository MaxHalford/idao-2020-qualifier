import collections
import os

import pandas as pd
import tqdm


SUBMISSIONS = ['ar']

SAT_VAR_MODELS = collections.defaultdict(lambda: collections.defaultdict(lambda: 'ar'))


if __name__ == '__main__':

    here = os.path.dirname(os.path.realpath(__file__))

    # The predictions will default to those of the auto-regressive model, we can thus manually
    # specify overrides here
    SAT_VAR_MODELS[0]['x'] = 'ar'

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
            blend.loc[ids, var] = subs[model].loc[ids, var]

    blend.to_csv(os.path.join(here, 'track_1_blended.csv'))
