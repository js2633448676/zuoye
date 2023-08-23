import pandas as pd
train_sets = pd.read_csv("data//train.csv")
train_sets.head()
train_sets.info()
train_sets.describe()
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.barplot(x="Sex",y="Survived",data=train_sets)
sns.barplot(x="Pclass",y="Survived", data=train_sets)
facet = sns.FacetGrid(train_sets, hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_sets['Age'].max()))
facet.add_legend()
sns.barplot(x="SibSp",y="Survived",data=train_sets)
sns.barplot(x="Parch",y="Survived",data=train_sets)
facet = sns.FacetGrid(train_sets, hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_sets['Fare'].max()))
facet.add_legend()
sns.barplot(x='Embarked',y='Survived',data=train_sets)
train_sets['Age'] = train_sets['Age'].fillna(train_sets['Age'].mean())
mode = train_sets['Embarked'].mode().values[0]
train_sets['Embarked'] = train_sets['Embarked'].fillna(mode)
features = ["Pclass", "Sex","Age", "SibSp", "Parch","Fare","Embarked"]
train_data = pd.get_dummies(train_sets[features])
train_label = train_sets['Survived']
train_X,test_X,train_y,test_y = train_test_split(train_data,train_label,test_size=0.2,random_state=666)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#使用KNN进行分类
knn_clf = KNeighborsClassifier()
knn_clf.fit(train_X, train_y)
print(knn_clf.score(test_X,test_y)) #out:0.6983
#使用逻辑回归进行分类
log_reg = LogisticRegression()
log_reg.fit(train_X, train_y)
print(log_reg.score(test_X,test_y)) #out:0.7765
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# 为逻辑回归添加多项式项的管道
def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])
# 使用管道得到对象
poly_log_reg = PolynomialLogisticRegression(degree=4)
poly_log_reg.fit(train_X, train_y)
print(poly_log_reg.score(test_X, test_y)) #out:0.8324
from sklearn.model_selection import cross_val_score
import numpy as np
scores = cross_val_score(poly_log_reg,train_X,train_y)
np.mean(scores) #0.8061
#测试数据预测
test_sets=pd.read_csv("test.csv")
test_data = test_sets[features]
test_data.info()
#缺失值处理(测试集缺失Age和Fare特征)
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
test_data.info()
#独热编码
test_data_v2 = pd.get_dummies(test_data)
test_data_v2.head()
#输出预测文件
predictions = poly_log_reg.predict(test_data_v2)
output = pd.DataFrame({'Survived': predictions})
output.to_csv('my_submission_2.csv', index=False)
print("Your submission was successfully saved!")

