from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


def logistic_regression(X, y):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))


if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    logistic_regression(X, y)