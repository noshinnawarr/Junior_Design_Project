from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import joblib

digits = load_digits()

model_params = {
    'svm' : {
        'model' : svm.SVC(),
        'params' : {
            'C' : [5],
            'kernel' : ['rbf'],
            'gamma' : ['scale']
        }
    }
}

scores = []
for model_name, mp in model_params.items():
  clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
  clf.fit(digits.data, digits.target)
  scores.append({
      'model_name' : model_name,
      'best_score' : clf.best_score_,
      'best_params' : clf.best_params_,
      'model' : clf
  })

model = scores[0]['model']
joblib.dump(model, 'model_joblib')
print(model.score(digits.data, digits.target))


