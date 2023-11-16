# Explain Decision Tree

## Overview
The Explain Decision Tree Package is a Python package designed to provide natural language explanations of predictions made by decision trees. This tool is useful for interpreting complex decision tree mechanisms, translating them into a format that is easier to understand for a wide range of users.

## Methodologies
The package incorporates two distinct methodologies for generating explanations, each catering to different aspects of decision tree interpretation:

### Rule-Based Decision Tree Explanations
The rule-based method focuses on creating explanations through a traversal of the decision tree along the path determined by the input features. It utilizes text-based templates to articulate the reasoning at each decision node in the tree. This approach directly correlates the tree's structural decisions with the resultant predictions, outlining the logic in a step-by-step manner.

### LLM-Based Decision Tree Explanations
The LLM-based approach leverages large language models to generate explanations. It involves translating key information from the decision path of the tree into a text prompt, which is then fed into an LLM. The LLM processes this prompt to produce a natural language explanation. This method aims to provide a more comprehensive narrative by incorporating a broader range of contextual information and insights into the tree's decision-making process.

## Virtual Environment and API Key Configuration

### Virtual Environment Requirement
The LLM-based decision tree explanations must be run within a virtual environment. This is necessary for the correct loading of environment variables from a `.env` file. The package's ability to locate and read the `.env` file depends on its installation path, which differs between a global installation and one within a virtual environment.

### API Key Setup
An OpenAI API key is required for the LLM-based explanations. This key needs to be stored in a `.env` file in your project's root directory. Follow these steps to set it up:

1. **Create a `.env` File**: In your project's root directory, create a `.env` file.

2. **Include the OpenAI API Key**:
   In the `.env` file, add your OpenAI API key as follows:
   ```
   API_KEY=your_openai_api_key_here
   ```
   Replace `your_openai_api_key_here` with your actual OpenAI API key.

Carefully manage the `.env` file and avoid including it in public version control systems to maintain the security of your API key.

## How to Use
To use either the rule-based or LLM-based explanation generator, you need to prepare a dictionary named `tree_data` with specific keys and then call the respective function.

### Preparing `tree_data`
Your `tree_data` dictionary should contain the following information:

- `'test_features'` (pd.DataFrame): The test feature data.
- `'test_labels'` (pd.Series): The test labels.
- `'classifier'` (DecisionTreeClassifier): The trained decision tree classifier.
- `'feature_names'` (List[str]): List of original feature names.
- `'categorical_columns'` (List[str]): List of names of categorical columns.
- `'original_data'` (pd.DataFrame): The original dataset.
- `'feature_description'` (Optional[Dict[str, str]]): Descriptions for the features, if available.
- `'data_index'` (int): Index of the specific test example to explain.

### Example
```python
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from explainDecisionTree import generate_rule_based_decision_tree_explanation, generate_llm_based_decision_tree_explanation

# Example data preparation
data = pd.read_csv('your_data.csv')
features = data.drop('target_column', axis=1)
labels = data['target_column']
clf = DecisionTreeClassifier().fit(features, labels)

tree_data = {
    'test_features': features,
    'test_labels': labels,
    'classifier': clf,
    'feature_names': list(features.columns),
    'categorical_columns': ['list', 'of', 'categorical', 'columns'],
    'original_data': data,
    'feature_description': {'feature1': 'description1', ...},
    'data_index': 0
}

generate_rule_based_decision_tree_explanation(tree_data) # rule-based explanation
generate_llm_based_decision_tree_explanation(tree_data) # llm-based explanation

```

Note that `prefix_path` is an optional parameter in both functions that allows you to specify a directory path where the explanations will be saved.

## Conclusion

Overall, the Explain Decision Tree package enhances the interpretability of decision trees, offering both rule-based and LLM-based explanations.




