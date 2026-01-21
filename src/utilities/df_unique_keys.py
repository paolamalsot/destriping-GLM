import pandas as pd

def enforce_unique_keys(df, unique_keys):
    if unique_keys:
        duplicates = df[df.duplicated(subset=unique_keys, keep=False)]
        if not duplicates.empty:
            raise ValueError(f"Duplicate entries found for keys: {unique_keys}.\n{duplicates}")


def _enforce_unique_keys(func):
    def wrapper(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        enforce_unique_keys(self.df, self.unique_keys)
        return res
    return wrapper


class UniqueKeyDF():

    def __init__(self, columns, unique_keys=[], dtypes = None):
        self.columns = columns
        self.dtypes = dtypes
        if dtypes is None:
            self.df = pd.DataFrame(columns=columns)
        else:
            self.df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in zip(columns, dtypes)})

        self.unique_keys = unique_keys

    @_enforce_unique_keys
    def append_dict(self, dict_row):
        ## the default behaviour is that if one of the unique keys already exists we replace it issueing a warning !

        # Convert the new row dictionary to a DataFrame
        new_df = pd.DataFrame.from_dict([dict_row])

        # Find intersection based on the unique keys
        intersection = pd.merge(new_df[self.unique_keys], self.df[self.unique_keys], how='inner')

        # Check if intersection is empty
        if not intersection.empty:
            keys_intersect = intersection[self.unique_keys].values.flatten().tolist()
            print(f"Warning: New keys {keys_intersect}. Erase old values.")

            # Drop the old conflicting rows
            self.df = self.df[~self.df[self.unique_keys].isin(keys_intersect).all(axis=1)]

        # Append the new row to the DataFrame
        if self.df.empty: #to avoid warning
            self.df = new_df
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)

        return self
    
    def copy(self):
        new_df = UniqueKeyDF(columns = self.columns, unique_keys=self.unique_keys, dtypes = self.dtypes)
        new_df.df = self.df.copy()
        return new_df
    
    @_enforce_unique_keys
    def save(self, file_path):
        # Save the unique keys as a metadata line
        # with open(file_path, 'w') as file:
        #     file.write(f"# unique_keys: {','.join(self.unique_keys)}\n")
        #     if self.dtypes is not None:
        #         file.write(f"# dtypes: {','.join(self.dtypes)}\n")
        
        # Append the DataFrame to the file in CSV format
        self.df.to_csv(file_path, index=False)

    def __repr__(self):
        return self.df.__repr__()
    
    @classmethod
    def load(cls, file_path):
        # # Read the unique keys from the first line
        # with open(file_path, 'r') as file:
        #     first_line = file.readline().strip()
        #     if first_line.startswith("# unique_keys:"):
        #         unique_keys = first_line.replace("# unique_keys:", "").split(',')
        #     else:
        #         unique_keys = []
        
        #     second_line = file.readline().strip()
        #     if second_line.startswith("# dtypes:"):
        #         dtypes = first_line.replace("# dtypes:", "").split(',')
        #     else:
        #         dtypes = None
        
        # Read the DataFrame from the CSV file (skipping the first line)
        df = pd.read_csv(file_path, comment='#')
        return cls.from_df(df)

    @classmethod
    def from_df(cls, df):
        instance = cls()
        instance.df = df
        instance.set_dtypes()
        instance.enforce_unique_keys()
        return instance
    
    @classmethod
    def from_records(cls, records):
        df = pd.DataFrame.from_records(records)
        return cls.from_df(df)

    def enforce_unique_keys(self):
        enforce_unique_keys(self.df, self.unique_keys)

    def set_dtypes(self):
        if self.dtypes is not None:
            columns = self.df.columns.tolist()
            dtypes_dict = {key: val for key, val in zip(columns, self.dtypes)}
            self.df = self.df.astype(dtypes_dict)


class img_df(UniqueKeyDF):

    dtypes = [object, bool, object, float]
    unique_keys = ["img_key"]

    def __init__(self):
        super().__init__(columns = ["img_key", "in_memory", "path", "scalefactor"], unique_keys= self.unique_keys, dtypes = self.dtypes)

    @_enforce_unique_keys
    def is_img_in_memory(self, img_key):
        query_str = "img_key == @img_key"
        return self.df.query(query_str)["in_memory"].any()

    @_enforce_unique_keys
    def add_img_path(self, img_key, img_path):
        query_str = "img_key == @img_key"
        index = self.df.query(query_str).index
        self.df.loc[index, "path"] = img_path

    @_enforce_unique_keys
    def remove_img_from_memory(self, img_key):
        query_str = "img_key == @img_key"
        index = self.df.query(query_str).index
        self.df.loc[index, "in_memory"] = False

    def add_img(self, img_key, path, scalefactor, in_memory):
        dict_row = {"img_key": img_key,
                "in_memory": in_memory,
                "path": path,
                "scalefactor": scalefactor}
        return self.append_dict(dict_row)

    @_enforce_unique_keys
    def get_img_path(self, img_key):
        query_str = "img_key == @img_key"
        res = self.df.query(query_str)["path"].item()
        return res

    @_enforce_unique_keys
    def get_img_scalefactor(self, img_key):
        query_str = "img_key == @img_key"
        res = self.df.query(query_str)["scalefactor"].item()
        return res

    def copy(self):
        res = super().copy()
        res_2 = img_df()
        res_2.df = res.df
        return res_2


class coordinates_df(UniqueKeyDF):

    dtypes = [object, object, float]
    unique_keys = ["coordinate_id", "img_key"]

    def __init__(self):
        super().__init__(columns = ["coordinate_id", "img_key", "scalefactor"], unique_keys= self.unique_keys, dtypes = self.dtypes)

    @_enforce_unique_keys
    def get_img_coordinate_dict(self, img_key, coordinate_id_key = None):
        if coordinate_id_key is None:
            query_str = "img_key == @img_key"
            return self.df.query(query_str).iloc[0].to_dict()
        else:
            query_str = "img_key == @img_key and coordinate_id == @coordinate_id"
            return self.df.query(query_str).iloc[0].to_dict()
        
    @_enforce_unique_keys
    def coordinates_sf_for_img(self, img_key):
        if img_key is not None:
            query_str = "img_key == @img_key"
            coordinates_df_series = self.df.query(query_str).iloc[0]

        else:
            mask = pd.isna(self.df["img_key"])
            coordinates_df_series = self.df.loc[mask,].iloc[0]

          # take the first one..
        return (
            coordinates_df_series["coordinate_id"],
            coordinates_df_series["scalefactor"],
        )
        

    
    def add_coordinates(self, coordinate_id, scalefactor, img_key):
        
        coordinates_dict = {
            "coordinate_id": coordinate_id,
            "img_key": img_key,
            "scalefactor": scalefactor
        }

        self.append_dict(coordinates_dict)

    def copy(self):
        res = super().copy()
        res_2 = coordinates_df()
        res_2.df = res.df
        return res_2
