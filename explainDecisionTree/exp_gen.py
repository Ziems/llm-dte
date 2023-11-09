import openai
from sklearn.tree import export_text
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def get_feature_descs(orig_feat_names, feature_desc_map):
    feature_desc = [f'{feature} represents the {feature_desc_map[feature]}' for feature in orig_feat_names if feature in feature_desc_map]
    feature_desc = ',\n'.join(feature_desc)
    return feature_desc



def print_tree_path(example, tree, feat_names, df_orig, cat_columns):
    left_child = tree.children_left
    right_child = tree.children_right

    node = 0  # start at the root
    path = []
    path_nodes = [node]
    relevant_features = []

    orig_sample = df_orig.loc[example.name]

    while left_child[node] != right_child[node]:  # while not a leaf node
        feature = tree.feature[node]
        threshold = tree.threshold[node]

        # Determine which child to go based on feature threshold
        if example[feature] <= threshold:
            decision = f"Feature {feat_names[feature]} <= {threshold:.2f} -> LEFT"
            path.append(decision)
            node = left_child[node]
        else:
            decision = f"Feature {feat_names[feature]} > {threshold:.2f} -> RIGHT"
            path.append(decision)
            node = right_child[node]

        path_nodes.append(node)
        feature_name = feat_names[feature]
        adjusted_feature_name = feature_name if '_'.join(feature_name.split('_')[:-1]) not in cat_columns else '_'.join(feature_name.split("_")[:-1])
        relevant_features.append({'name': adjusted_feature_name, 'value': orig_sample[adjusted_feature_name]})

    # Print the decisions taken
    path_str = ""
    for step, decision in enumerate(path, 1):
        path_str += f"Step {step}: {decision}" + '\n'

    relevant_feature_str = ""
    for feature in relevant_features:
        relevant_feature_str += f"{feature['name']} = {feature['value']}" + '\n'

    return path_str, relevant_feature_str


def get_hydrated_prompt(example, orig_feat_names, feat_names, clf, df_orig, cat_columns, label, labels, feature_desc_map, prompt_id='a'):

    feature_desc = get_feature_descs(orig_feat_names, feature_desc_map)
    tree_text = export_text(clf, feature_names=feat_names)
    path_str, relevant_feature_str = print_tree_path(example, clf.tree_, feat_names, df_orig, cat_columns)

    prompt = f"""Suppose a dataset for network intrusion detection has the following features:
    {feature_desc}

    The labels are {' and '.join(labels)}.

    The following decision tree was build using the above features:
    {tree_text}

    A new test example has the following relevant features:
    {relevant_feature_str}

    The new test example took the following path through the tree:
    {path_str}

    Using inferred background knowledge of the features and network traffic, explain in simple terms why the decision tree came to the conclusion that the given example is {label}.
    Do not refer to the underlying mechanics of the decision tree in any way, and only refer to the features using natural language. Please refer to the feature values in context using parenthesis.
    """
    return prompt


def generate_llm_explanation(example, original_feat_names, feat_names, clf, df_orig, cat_columns, label, labels, feature_desc_map, prompt_id='a'):
    prompt = get_hydrated_prompt(example, original_feat_names, feat_names, clf, df_orig, cat_columns, label, labels, feature_desc_map, prompt_id=prompt_id)
    api_key = os.getenv('API_KEY')
    openai.api_key = api_key

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def generate_rulebased_explanation(example, original_feat_names, feat_names, clf, df_orig, cat_columns, labels, feature_desc_map, prompt_id='a'):
    feature_desc = get_feature_descs(original_feat_names, feature_desc_map)
    path_str, relevant_feature_str = print_tree_path(example, clf.tree_, feat_names, df_orig, cat_columns)

    prompt = f"""Suppose a dataset for network intrusion detection has the following features:
    {feature_desc}

    The labels are {' and '.join(labels)}.

    A new test example has the following relevant features:
    {relevant_feature_str}

    The new test example took the following path through the tree:
    {path_str}
    """
    return prompt
