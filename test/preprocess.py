import numpy as np
import pandas as pd
import os, json

def features_from_sensitivewords(dataframe, sensitive_words):
    dataframe = pd.concat(
        [
            dataframe, 
            dataframe['title'].apply(lambda x: pd.Series({w: 1 if w in x else 0 for w in sensitive_words}))
        ],
        axis=1
    )
    return dataframe

def feature_engineering(dataframe):
    dataframe["durationType"] = pd.cut(
        dataframe["duration"], 
        [
            -float("inf"), 
            30, 
            60, 
            180, 
            300, 
            float("inf")
        ],
        labels=["extraShort", "short", "medium", "long", "extraLong"]
    ).astype(str)
    return dataframe

def keep_topK(dataframe, feat_name, value_dict, topK):
    new_value_dict = {}
    topK_values = dataframe[feat_name].value_counts().head(topK).index
    for val in topK_values:
        new_value_dict[val] = value_dict[val]
    return new_value_dict

def calc_featureXfeature_killRate(
    dataframe, 
    feat_names,
    save_path,
    topK=None, 
    at_least=3
):
        featureXfeature = "X".join(feat_names)
        if len(feat_names) > 1:
            dataframe[featureXfeature] = dataframe[feat_names[0]]
            for name in feat_names[1:]:
                dataframe[featureXfeature] += "_" + dataframe[name]
        rate = dataframe.groupby(featureXfeature).apply(lambda x: (x["label"] == 1).mean())
        dataframe[f'{featureXfeature}KillRate'] = dataframe[featureXfeature].map(rate)
        rate_dict = rate.to_dict()
        if not topK is None:
            rate_dict = keep_topK(dataframe, featureXfeature, rate_dict, topK)
        if not at_least is None:
            counts = dataframe[featureXfeature].value_counts()
            discard = counts[counts < at_least].index
            for k in discard:
                rate_dict.pop(k)
        rate_dict["未知"] = (dataframe["label"] == 1).sum() / len(dataframe)
        print(f"{len(rate_dict)} items are in `{featureXfeature}KillRateDict`")
        with open(save_path + os.sep + f'{featureXfeature}KillRate.json', "w") as f:
            json.dump(rate_dict, f)
        return featureXfeature, f'{featureXfeature}KillRate'

def calc_killRate(
    dataframe,
    feat_combinations,
    save_path,
    topK=None,
    at_least=None
):
        featXfeat_names = []
        featXfeat_rate_names = []
        for pair in feat_combinations:
            name, rate_name = calc_featureXfeature_killRate(
                dataframe, 
                pair,
                save_path,
                topK, 
                at_least
            )
            featXfeat_names.append(
                name
            )
            featXfeat_rate_names.append(
                rate_name
            )
        file_path = save_path + os.sep + "killRate_feature_combinations.json"
        print(f"Saving feature combinations to {file_path}")
        with open(file_path, "w") as f:
            json.dump(feat_combinations, f)
        return featXfeat_names, featXfeat_rate_names

def auto_load_killRate(save_path):
    with open(save_path + os.sep + "killRate_feature_combinations.json", "r") as f:
        feat_combinations = json.load(f)
    rate_series = []
    for pair in feat_combinations:
        file_name = "X".join(pair) + "KillRate.json"
        with open(save_path + os.sep + file_name, "r") as f:
            killRate_dict = json.load(f)
        rate_series.append(pd.Series(killRate_dict))
    return feat_combinations, rate_series

def fill_featureXfeature_killRate(
    dataframe, 
    feat_names, 
    rate_series
):
        featureXfeature = "X".join(feat_names)
        dataframe[featureXfeature] = dataframe[feat_names[0]]
        for name in feat_names[1:]:
            dataframe[featureXfeature] += "_" + dataframe[name]
        dataframe[f'{featureXfeature}KillRate'] = dataframe[featureXfeature].map(
            rate_series
        ).fillna(rate_series["未知"])

def auto_fill_killRate(dataframe, feat_combinations, rate_series):
    for pair, series in zip(feat_combinations, rate_series):
        fill_featureXfeature_killRate(dataframe, pair, series)