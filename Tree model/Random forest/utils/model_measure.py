import sklearn
import sklearn.metrics

def measure_regression(y_true, y_pred):


    score = {
        'mae'  : sklearn.metrics.mean_absolute_error(y_true, y_pred),
        'mse'  : sklearn.metrics.mean_squared_error(y_true, y_pred),
        'mape' : sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred),
        'r2'   : sklearn.metrics.r2_score(y_true, y_pred) }

    return score
# =============================================================================
def print_regression_measure(y_true, y_pred, digit=3):

    score = measure_regression(y_true, y_pred)

    for k, v in score.items():
        k = k.upper()
        v = round(v, digit)
        print(f'{k:>4} : {v}')
# =============================================================================
def get_measure_name(task):
    if task in ['regression', 'reg', 'r']:
        name_li = ['mse', 'rmse', 'mae', 'mape', 'r2']
    elif task in ['classification', 'clf', 'c']:
        name_li = []
    return name_li
# =============================================================================
def get_regression_metric_function(name):
    if name.lower() in ['mae']:
        func = sklearn.metrics.mean_absolute_error
    elif name.lower() in ['mse']:
        func = sklearn.metrics.mean_squared_error
    elif name.lower() in ['mape']:
        func = sklearn.metrics.mean_absolute_percentage_error
    elif name.lower() in ['r2']:
        func = sklearn.metrics.r2_score
# =============================================================================