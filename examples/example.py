# Example of the llm-based Decision tree explanations using the sklearn Iris dataset.

from explainDecisionTree import generate_llm_based_decision_tree_explanation
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
y = y.replace([0, 1, 2], iris.target_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

tree_data = {
    'test_features': X_test,
    'test_labels': y_test,
    'classifier': clf,
    'feature_names': iris.feature_names,
    'categorical_columns': [],  # Iris dataset does not have categorical features
    'original_data': X,
    'data_index': 5  # random data index for demonstration
}

generate_llm_based_decision_tree_explanation(tree_data)
