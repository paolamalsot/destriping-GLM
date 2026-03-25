from cmap import Colormap

cm = Colormap("okabeito:okabeito")
colorblind_palette = cm.to_mpl().colors

model_name_replacement_dict = {
    "2SN": "b2c",
    "bin_level_normalization": "bin-level norm.",
    "dividing_99_quantile": "b2c-sym",
    "dividing_99_quantile_nucl": "b2c-sym-nucl",
    "dividing_median": "b2c-sym-med",
    "dividing_median_quantile_nucl": "b2c-sym-med-nucl",
    "dividing_median_ratio": "MRSE",
    "ours__none_init": "ours_N.I",
    "ours__P2_identity": "ours_P2.I",
    "GT_nbinom_sol": "GT",
    "collapse_label": "ours (collapse label)",
}

color_dict = {
    "b2c": colorblind_palette[6],
    "b2c-sym": colorblind_palette[7],
    "b2c-sym-nucl": colorblind_palette[7],
    "bin-level norm.": colorblind_palette[1],
    "original": colorblind_palette[0],
    "ours": colorblind_palette[3],
    "ours (collapse label)": colorblind_palette[5],
    "ours_N.I": colorblind_palette[3],
    "ours_P2.I": colorblind_palette[3],
    "GT": colorblind_palette[4],
    "MRSE": colorblind_palette[2],
    "b2c-sym-med": colorblind_palette[5],
    "b2c-sym-med-nucl": colorblind_palette[5],
    "b2c-sym-c_mean": colorblind_palette[2],
}
