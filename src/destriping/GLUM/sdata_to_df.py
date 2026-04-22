def format_obs_to_df(df, cell_id_label):
    df = df.rename(columns={"array_row": "i", "array_col": "j", cell_id_label: "p"})
    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)
    df["p"] = df["p"].astype(object)
    df["k"] = df["n_counts"].astype(int)
    # for c in ["i", "j", "p"]:
    #    df[c] = df[c].cat.remove_unused_categories()
    return df[["i", "j", "p", "k"]]


def format_data_to_obs(data):
    data.n_counts  # just to mirror your original side-effect
    data.add_array_coords_to_obs()
    return data.obs


def data_to_df(data, id_label):
    data.add_array_coords_to_obs()
    obs = data.obs
    obs = obs.query(
        f"~{id_label}.isna()"
    ).copy()  # this will NOT erase the cells "NA" "NaN" ?
    obs[id_label] = obs[id_label].astype(str)
    obs[id_label] = "id_" + obs[id_label]
    df = format_obs_to_df(obs, id_label)
    return df
