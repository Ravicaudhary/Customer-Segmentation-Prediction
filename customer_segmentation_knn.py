import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Age":[22,25,28,30,32,35,38,40,42,45,48,50,52,55,58,60,23,27,31,34,37,41,44,47,51,54,57,61,26,33],
    "Annual_Income":[25000,28000,30000,32000,35000,38000,40000,42000,45000,48000,50000,52000,55000,58000,60000,65000,27000,31000,34000,37000,41000,46000,49000,53000,57000,62000,68000,72000,29000,36000],
    "Spending_Score":[20,25,30,28,35,32,40,38,45,42,50,48,55,52,60,58,22,27,33,36,39,44,47,51,54,59,63,68,26,34],
    "Visits_Per_Month":[2,3,3,4,4,5,5,6,6,7,7,8,8,9,10,10,2,3,4,4,5,6,6,7,8,9,10,11,3,4],
    "Segment":["Low Value","Low Value","Low Value","Low Value","Low Value","Low Value","Low Value","Low Value","Regular","Regular","Regular","Regular","Regular","Regular","High Value","High Value","Low Value","Low Value","Low Value","Low Value","Regular","Regular","Regular","Regular","Regular","High Value","High Value","High Value","Low Value","Low Value"]
}
df = pd.DataFrame(data)
X = df[["Age","Annual_Income","Spending_Score","Visits_Per_Month"]]
y = df["Segment"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results=[]
for k in range(1,12):
    model=KNeighborsClassifier(n_neighbors=k,weights="distance")
    model.fit(X_train_scaled,y_train)
    pred=model.predict(X_test_scaled)
    results.append({"K":k,"Recall":recall_score(y_test,pred,average="weighted")})

results_df=pd.DataFrame(results)
best_k=int(results_df.loc[results_df["Recall"].idxmax(),"K"])
knn=KNeighborsClassifier(n_neighbors=best_k,weights="distance")
knn.fit(X_train_scaled,y_train)
y_pred=knn.predict(X_test_scaled)

print("Best K:",best_k)
print("Accuracy:",round(accuracy_score(y_test,y_pred),3))
print("Precision:",round(precision_score(y_test,y_pred,average="weighted"),3))
print("Recall:",round(recall_score(y_test,y_pred,average="weighted"),3))
print("F1 Score:",round(f1_score(y_test,y_pred,average="weighted"),3))
print("\nClassification Report:\n",classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred,labels=["High Value","Regular","Low Value"])
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["High Value","Regular","Low Value"],yticklabels=["High Value","Regular","Low Value"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("KNN Customer Segmentation Confusion Matrix"); plt.show()

new_customer=pd.DataFrame({"Age":[32],"Annual_Income":[90000],"Spending_Score":[88],"Visits_Per_Month":[17]})
prediction=knn.predict(scaler.transform(new_customer))
print("\nNew Customer Predicted Segment:",prediction[0])
