from .ques_gen import generate_questions
from .exp_gen import generate_llm_explanation, generate_rulebased_explanation


def generate_rule_based_decision_tree_explanation(tree_data, prefix_path=None):
    _process_decision_tree_examples(tree_data, prefix_path, use_llm=False)


def generate_llm_based_decision_tree_explanation(tree_data, prefix_path=None):
    _process_decision_tree_examples(tree_data, prefix_path, use_llm=True)


def _process_decision_tree_examples(tree_data, prefix_path, use_llm):
    X_test = tree_data['X_test']
    y_test = tree_data['y_test']
    clf = tree_data['clf']
    orig_feature_names = tree_data['orig_feature_names']
    cat_columns = tree_data['cat_columns']
    df_orig = tree_data['df_orig']
    feature_desc_map = tree_data['feature_desc_map']
    index = tree_data['df_index']
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

    # save examples
    if not prefix_path:
        prefix_path = 'llm' if use_llm else 'rule-based'
    with open(f'exps/{prefix_path}/example_{index}.txt', 'w') as f:
        f.write(explanation)
        f.write('\n\n\n')
        for q in questions:
            f.write(q['q'] + '\n')
            f.write(str(q['a']) + '\n\n')
