from sklearn.metrics import fbeta_score, make_scorer, matthews_corrcoef


SCORERS = {
    "F2": make_scorer(fbeta_score, beta=2),
    # "MCC": make_scorer(matthews_corrcoef)
}


DEF_SCORER = SCORERS['F2']