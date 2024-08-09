import copy
import sklearn
import sklearn.base
import sklearn.ensemble


def training_random_forest_early_stop(
    data_X, data_Y, sample_weight = None,
    max_estimators = 100,
    n_iter_no_change = 20,
    refit = True,
    return_logs = False,
    **kwargs):
    '''
    find optimal "n_estimators" based on early stop method on the oob score

    parameters:
    data_X (2d numpy.ndarray) : input data
    data_Y (1d numpy.ndarray) : output data
    **kwargs : arguments passed to RandomForestRegressor()
    '''

    # logs
    oob_score = []
    model_li  = []

    # initialization
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=1, warm_start=False, **kwargs)
    model.fit(data_X, data_Y, sample_weight=sample_weight)

    # best model
    best_model = copy.deepcopy(model)
    n_iter = 1

    # add logs
    oob_score.append(model.oob_score_)

    for n_estimators in range(2, max_estimators+1):
        # refit model
        model.set_params(n_estimators=n_estimators, warm_start=True)
        model.fit(data_X, data_Y, sample_weight=sample_weight)

        # score, smaller is better
        if best_model.oob_score_ > model.oob_score_:
            best_model = copy.deepcopy(model)
            n_iter = 1
        else:
            n_iter = n_iter + 1

        # add logs
        oob_score.append(model.oob_score_)

        # reach the maximal no changes
        if n_iter >= n_iter_no_change:
            break

    else:
        best_model = sklearn.base.clone(model)

    # reset 'warm_start=False' for refitting model
    best_model.set_params(warm_start=False)

    # refitting the whole model
    if refit:
        best_model = sklearn.base.clone(best_model)
        best_model.fit(data_X, data_Y, sample_weight=sample_weight)

    if return_logs:
        return best_model, oob_score
    else:
        return best_model
# =========================================================================