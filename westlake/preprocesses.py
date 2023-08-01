from collections import defaultdict


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