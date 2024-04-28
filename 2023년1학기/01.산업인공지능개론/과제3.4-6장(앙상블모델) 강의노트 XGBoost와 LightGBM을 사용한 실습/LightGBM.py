from lightgbm import LGBMClassifier, plot_importance, plot_metric, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_validate
import lightgbm as lgb

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=123)
lgbmc = LGBMClassifier(n_estimators=400)
evals = [(X_test, y_test)]
early_stopping = lgb.early_stopping(stopping_rounds=100, first_metric_only=True)
log_callback = lgb.log_evaluation(period=10, show_stdv=True)
lgbmc.fit(X_train, y_train, eval_set=evals, eval_metric='logloss', callbacks=[early_stopping, log_callback])
preds = lgbmc.predict(X_test)

cross_val = cross_validate(estimator=lgbmc, X=iris.data, y=iris.target, cv=5)
print('avg fit time: {} (+/- {})'.format(cross_val['fit_time'].mean(), cross_val['fit_time'].std()))
print('avg score time: {} (+/- {})'.format(cross_val['score_time'].mean(), cross_val['score_time'].std()))
print('avg test score: {} (+/- {})'.format(cross_val['test_score'].mean(), cross_val['test_score'].std()))

plot_metric(lgbmc)
plot_importance(lgbmc, figsize=(10, 12))
plot_tree(lgbmc, figsize=(28, 24))
