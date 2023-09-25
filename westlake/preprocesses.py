from collections import defaultdict

import numpy as np
import pandas as pd


def prepare_specie_table(df_reac, df_spec=None):
    """Prepare specie table.

    This function collect all species in the reaction network, and create a
    specie table. The output will always have an index column (0, 1, 2, ...).
    The output will include the columns in ``df_spec`` if it is provided.

    Args:
        df_reac (pd.DataFrame): Reaction network.
        df_spec (pd.DataFrame, optional): Base specie table. Defaults to None.

    Returns:
        pd.DataFrame: Specie table.
    """
    species = set(df_reac["reactant_1"].values)
    species.update(set(df_reac["reactant_2"].values))
    for prods in df_reac["products"].values:
        species.update(set(prods.split(";")))
    species.remove("") # Remove empty species
    specie_list = list(species)
    specie_list.sort()

    if df_spec is None:
        return pd.DataFrame(index=specie_list)

    species_test = set(df_spec.index.values)
    if not species.issubset(species_test):
        raise ValueError("Mising species in the specie table: {}.".format(
            species.difference(species_test).__repr__()))

    df_spec_new = df_spec.copy()
    df_spec_new = df_spec_new.loc[specie_list]
    inds = np.arange(len(specie_list))
    return df_spec_new


def prepare_piecewise_rates(df_reac):
    """Prepare variables to deal with piecewise rates."""
    # Put the same reactions into a dict, and find the min/max temperature.
    temp_min_dict = defaultdict(list)
    temp_max_dict = defaultdict(list)
    for idx, (key, temp_min, temp_max) in enumerate(zip(
        df_reac["key"].values, df_reac["T_min"].values, df_reac["T_max"].values
    )):
        temp_min_dict[key].append((temp_min, idx))
        temp_max_dict[key].append((temp_max, idx))

    is_leftmost = [False]*len(df_reac)
    for temps in temp_min_dict.values():
        temp_min = min(temps, key=lambda x: x[0])
        is_leftmost[temp_min[1]] = True

    is_rightmost = [False]*len(df_reac)
    for temps in temp_max_dict.values():
        temp_max = max(temps, key=lambda x: x[0])
        is_rightmost[temp_max[1]] = True

    df_reac["is_leftmost"] = is_leftmost
    df_reac["is_rightmost"] = is_rightmost