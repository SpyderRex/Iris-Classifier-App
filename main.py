from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

model = LogisticRegression()
model.fit(X, y)

print("Greetings. This simple app allows you to predict the type of an iris - Setosa, Versicolor, or Virginica - from the sepal length and the sepal width. A sepal is a part of a flower and typically functions as protection for the flower in bud, and often as support for the petals when in bloom.")

print()

sepal_length = float(input("Enter sepal length in cm: "))
sepal_width = float(input("Enter sepal width in cm: "))
prediction = model.predict([[sepal_length, sepal_width]])

print()


if prediction == 0:
    print("Our model has predicted that the type of iris with these specifications is Setosa.")
elif prediction == 1:
    print("Our model has predicted that the type of iris with these specifications is Versicolor.")
else:
    print("Our model has predicted that the type of iris with these specifications is Virginica.")
    
print()
    
accuracy = model.score(X,y)
print(f"This predictive model is {accuracy:.1%} accurate.")