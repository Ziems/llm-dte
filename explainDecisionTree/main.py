import os
from typing import Optional, Dict, Any
from explainDecisionTree.ques_gen import generate_questions
from explainDecisionTree.exp_gen import generate_llm_explanation, generate_rulebased_explanation


def generate_rule_based_decision_tree_explanation(tree_data: Dict[str, Any], prefix_path: Optional[str] = None) -> None:
    """
    Generates an explanation based on a decision tree using a rule-based approach.

    Parameters:
    tree_data (Dict[str, Any]): A dictionary containing the following keys:
        - 'test_features' (pd.DataFrame): The test feature data.
        - 'test_labels' (pd.Series): The test labels.
        - 'classifier' (DecisionTreeClassifier): The trained decision tree classifier.
        - 'feature_names' (List[str]): List of original feature names.
        - 'categorical_columns' (List[str]): List of names of categorical columns.
        - 'original_data' (pd.DataFrame): The original dataset.
        - 'feature_description' (Optional[Dict[str, str]]): Descriptions for the features, if available.
        - 'data_index' (int): Index of the specific test example to explain.
    prefix_path (Optional[str]): Optional path prefix for saving the explanation.
    """
    explanation, questions = _process_decision_tree_examples(tree_data, use_llm=False)
    _save_explanation_and_questions(explanation, questions, tree_data['data_index'], prefix_path, use_llm=False)


def generate_llm_based_decision_tree_explanation(tree_data: Dict[str, Any], prefix_path: Optional[str] = None) -> None:
    """
    Generates an explanation based on a decision tree using a large language model (LLM) approach.

    Parameters:
    tree_data (Dict[str, Any]): A dictionary containing the data needed for generating the explanation, structured as in generate_rule_based_decision_tree_explanation.
    prefix_path (Optional[str]): Optional path prefix for saving the explanation.
    """
    explanation, questions, = _process_decision_tree_examples(tree_data, use_llm=True)
    _save_explanation_and_questions(explanation, questions, tree_data['data_index'], prefix_path, use_llm=True)


def _process_decision_tree_examples(tree_data, use_llm):
    X_test = tree_data['test_features']
    y_test = tree_data['test_labels']
    clf = tree_data['classifier']
    orig_feature_names = tree_data['feature_names']
    cat_columns = tree_data['categorical_columns']
    df_orig = tree_data['original_data']
    feature_desc_map = tree_data.get('feature_description', None)
    index = tree_data['data_index']
    example = X_test.iloc[index]
    feature_names = X_test.columns.values
    label = y_test.iloc[index]
    labels = y_test.unique()

    if use_llm:
        explanation = generate_llm_explanation(
            example,
            orig_feature_names,
            feature_names,
            clf,
            df_orig,
            cat_columns,
            label,
            labels,
            feature_desc_map
        )
    else:
        explanation = generate_rulebased_explanation(
            example,
            orig_feature_names,
            feature_names,
            clf,
            df_orig,
            cat_columns,
            labels,
            feature_desc_map
        )

    questions = generate_questions(
        example,
        clf.tree_,
        feature_names,
        cat_columns,
        orig_feature_names,
    )

    return explanation, questions


def _save_explanation_and_questions(explanation, questions, index, prefix_path, use_llm):
    if not prefix_path:
        prefix_path = 'llm' if use_llm else 'rule-based'

    # Create directory if it doesn't exist
    dir_path = f'exps/{prefix_path}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(f'{dir_path}/example_{index}.txt', 'w') as f:
        f.write(explanation)
        f.write('\n\n\n')
        for q in questions:
            f.write(q['q'] + '\n')
            f.write(str(q['a']) + '\n\n')
