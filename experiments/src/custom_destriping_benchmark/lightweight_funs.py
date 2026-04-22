import os


def save_n_counts_adjusted_counts(data, output_dir, supp_columns=None):
    data.n_counts
    os.makedirs(output_dir, exist_ok=True)
    cols_to_save = ["n_counts"]
    if "n_counts_adjusted" in data.obs.columns:
        cols_to_save.append("n_counts_adjusted")
    if supp_columns is not None:
        cols_to_save.extend(supp_columns)

    to_save = data.obs[cols_to_save]
    out_path = os.path.join(output_dir, "df.parquet")
    to_save.to_parquet(out_path, compression="snappy")
