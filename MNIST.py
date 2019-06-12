from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X,y= mnist["data"], mnist['target']
X.shape
y.shape

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
plt.axis("off")
y=y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5=(y_train==5)
y_test_5=(y_test==5)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([X[0]])

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, scoring = 'accuracy', cv=10)

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
confusion_matrix(y_train_5, y_train_5)

#Precision & Recall
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

#F1_Score
from sklearn.metrics import f1_score
f1_score(y_train_5,y_train_pred)

y_scores = sgd_clf.decision_function([some_digit])
y_scores

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method = "decision_function")


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label = "Recall")
    plt.legend()
    plt.show()
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

threshold_90_precision = thresholds[np.argmax(precisions>=0.90)]
threshold_90_precision
y_train_pred_90 = (y_scores>=threshold_90_precision)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

from sklearn.metrics import roc_curve
fpr, tpr,thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr,label = None):
    plt.plot(fpr,tpr, linewidth=2, label = label)
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    
plot_roc_curve(fpr,tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method = 'predict_proba')

y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label = "SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc = "lower right")
plt.show()

#Multiclass Classification
#Stochastic Gradient Descent Classifier
sgd_clf.fit(X_train, y_train)
y_train_pred = sgd_clf.predict(X_train)
confusion_matrix(y_train, y_train_pred)

some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores

#Random Forest  Classifier
forest_clf.fit(X_train,y_train)
forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring= "accuracy")

#Featuring Scaling and Calculating Cross Validation Score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring= "accuracy")
#Confusion Matrix
conf_mx = confusion_matrix(y_train, y_train_pred)
#Visualising Confusion Matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()