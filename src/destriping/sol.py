from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd
if TYPE_CHECKING:
    from src.spatialAdata.spatialAdata import spatialAdata
from src.spatialAdata.loading import load_spatialAdata
import os

class Sol():
    def __init__(self, h: pd.Series, w: pd.Series, c: spatialAdata):
		# c is supposed to be bin expression data.
        self.w = w
        self.h = h
        self.c = c

    def save_h(self, output_dir):
        path = os.path.join(output_dir, "h.csv")
        self.h.to_csv(path)

    def save_w(self, output_dir):
        path = os.path.join(output_dir, "w.csv")
        self.w.to_csv(path)

    def save_c(self, output_dir):
        adata_path = os.path.join(output_dir, "c")
        self.c.save(adata_path) # saves the anndata	
    
    def save(self, output_dir):
        self.save_h(output_dir)
        self.save_w(output_dir)
        self.save_c(output_dir)

    @classmethod
    def load_h(cls, input_dir):
        path = os.path.join(input_dir, "h.csv")
        return pd.read_csv(path, index_col=0).iloc[:,0]

    @classmethod
    def load_w(cls, input_dir):
        path = os.path.join(input_dir, "w.csv")
        return pd.read_csv(path, index_col=0).iloc[:,0]

    @classmethod
    def load_c(cls, input_dir):
        adata_path = os.path.join(input_dir, "c")
        return load_spatialAdata(adata_path)
    
    @classmethod
    def load(cls, input_dir):
        h = cls.load_h(input_dir)
        w = cls.load_w(input_dir)
        c = cls.load_c(input_dir)
        return Sol(h, w, c)
