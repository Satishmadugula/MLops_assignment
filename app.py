import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import mlflow
import dagshub
import matplotlib.pyplot as plt


datapath = "winequality-red.csv"
dagshub.init(repo_owner='Satishmadugula', repo_name='MLops_assignment', mlflow=True)
mlflow.set_experiment("Wine Quality Prediction")




df = pd.read_csv(datapath, sep=',', header=0)

print(df.head())

# # Check if each column has any NaN values (True/False)
# print(df.isna().any())
# # Count NaN values in each column
# print(df.isna().sum())

# # Check if there are any NaNs in the entire DataFrame
# print(df.isna().values.any())

print("*"*30)
X = df.drop("quality", axis=1)
y = df["quality"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#imputing
fil = SimpleImputer(strategy="median")
X_train = fil.fit_transform(X_train)
X_test = fil.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the fitted scaler to a pickle file
scaler_filename = 'Wine_quality_scaler.pkl'
pickle.dump(scaler, open(scaler_filename, 'wb'))
mlflow.log_artifact(scaler_filename, "Wine_quality_scaler")

print(X_train.shape, X_test.shape)

# # voting classifier

# clf1 = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42)
# clf2 = GradientBoostingClassifier(n_estimators=15, learning_rate=0.002, random_state=42)
# clf3 = XGBClassifier(n_estimators=55, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric="logloss")

# voting_clf = VotingClassifier(
#     estimators=[("rf", clf1), ("gb", clf2), ("xgb", clf3)],
#     voting="soft"
# )
# with mlflow.start_run(run_name="Wine Quality VotingClassifier_RF_GB_XGB"):

#     # Train Voting Classifier
#     voting_clf.fit(X_train, y_train)
#     y_pred = voting_clf.predict(X_test)

#     y_probs = voting_clf.predict_proba(X_test)[:, 1]
#     # Evaluate metrics
#     acc = accuracy_score(y_test, y_pred)
#     report_dict = classification_report(y_test, y_pred, output_dict=True)
    
#     # Log params of VotingClassifier
#     mlflow.log_param("voting", "soft")
#     mlflow.log_param("estimators", ["RandomForest", "GradientBoosting", "XGBoost"])

#     # Log individual base model params
#     mlflow.log_params({f"rf_{k}": v for k, v in clf1.get_params().items()})
#     mlflow.log_params({f"gb_{k}": v for k, v in clf2.get_params().items()})
#     mlflow.log_params({f"xgb_{k}": v for k, v in clf3.get_params().items()})

#     # Log metrics
#     mlflow.log_metric("accuracy", acc)
#     mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
#     mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])
#     mlflow.log_metric("f1_score_macro", report_dict["macro avg"]["f1-score"])
#     mlflow.log_metric('recall_class_3', report_dict['3']['recall'])
#     mlflow.log_metric('recall_class_4', report_dict['4']['recall'])
#     mlflow.log_metric('recall_class_5', report_dict['5']['recall'])
#     mlflow.log_metric('recall_class_6', report_dict['6']['recall'])
#     mlflow.log_metric('recall_class_7', report_dict['7']['recall'])
#     mlflow.log_metric('recall_class_8', report_dict['8']['recall'])
    
#     # Save model as artifact
#     filename = r'Wine_quality_voting_classifier.pkl'
#     pickle.dump(voting_clf, open(filename, 'wb'))
#     #save as articfact
#     mlflow.log_artifact(r"Wine_quality_voting_classifier.pkl", artifact_path="models")




## random forest
params = {
    "n_estimators":35,
    "criterion": "gini",
    "random_state": 42,
}
model = RandomForestClassifier(**params)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)
#print(classification_report(y_pred,y_test))
report_dict = classification_report(y_test, y_pred, output_dict=True)
print("*"*75)
print("*"*75)
print(report_dict.keys())
print("*"*75)
print("*"*75)
with mlflow.start_run():
    mlflow.set_tag("author","Satish")
    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_3': report_dict['3']['recall'],
        'recall_class_4': report_dict['4']['recall'],
        'recall_class_5': report_dict['5']['recall'],
        'recall_class_6': report_dict['6']['recall'],
        'recall_class_7': report_dict['7']['recall'],
        'recall_class_8': report_dict['8']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    filename = r'Wine_quality_Random_forest_classifier.pkl'
    pickle.dump(model, open(filename, 'wb'))
    # Log the model file as an artifact
    mlflow.log_artifact(filename, "Wine Quality Randome Forest Model")
print(report_dict)



# ##n xgboost

# # ## XGBClassifier

# le = LabelEncoder()
# y_train_enc = le.fit_transform(y_train)   # maps {3,4,5,6,7,8} -> {0..5}
# y_test_enc  = le.transform(y_test)

# num_classes = len(le.classes_)  # 6

# params = {
#     "learning_rate":0.1,
#  "n_estimators":1000,
#  "max_depth":5,
#  "min_child_weight":1,
#  "gamma":0,
#  "subsample":0.8,
#  "colsample_bytree":0.8,
#  "objective": 'multi:softprob',
#  "nthread":4,
#  "scale_pos_weight":1,
#  "seed":27
# }
# model = XGBClassifier(**params)

# model.fit(X_train,y_train_enc)

# y_pred_enc = model.predict(X_test)
# y_pred = le.inverse_transform(y_pred_enc)

# print(classification_report(y_pred,y_test))
# report_dict = classification_report(y_test, y_pred, output_dict=True)
# # print("*"*75)
# # print("*"*75)
# # print(report_dict.keys())
# # print("*"*75)
# # print("*"*75)
# with mlflow.start_run():
#     mlflow.set_tag("author","Satish")
#     mlflow.log_params(params)
#     mlflow.log_metrics({
#         'accuracy': report_dict['accuracy'],
#         'f1_score_macro': report_dict['macro avg']['f1-score']
#     })
#   # Log per-class recall dynamically using original class names
#     for cls in le.classes_:
#         cls_key = str(cls)
#         if cls_key in report_dict:
#             mlflow.log_metric(f"recall_class_{cls_key}", report_dict[cls_key]["recall"])

#     filename = r'Wine_quality_XGBOOST_classifier.pkl'
#     pickle.dump(model, open(filename, 'wb'))
#     # Log the model file as an artifact
#     mlflow.log_artifact(filename, "Wine_quality_XGBOOST_classifier")
# print(report_dict)


