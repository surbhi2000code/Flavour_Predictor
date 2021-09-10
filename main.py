import pandas
from sklearn.tree import DecisionTreeClassifier, LabelEncoder

dt = pandas.read_csv('flavour.csv')
dt

Enc = LabelEncoder()
Enc.fit(['Male', 'Female'])
dt['Gender'] = Enc.transform(dt['Gender'])

X = dt.drop(columns=['Flavour'])
y = dt.drop(columns=['Age', 'Gender'])

CModel = DecisionTreeClassifier()
CModel.fit(X, y)

age = 18
gender = Enc.transform(['Male'])
CModel.predict([ [age, gender] ])

def flav_pred():
    age = int(input('Age:'))
    gen = input('Gender:').capitalize()
    gender = Enc.transform([gen])
    flav = CModel.predict([[age, gender]])
    print('Recommended Flavour:', flav[0])
flav_pred()