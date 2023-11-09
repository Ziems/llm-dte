import random


def original_feature_name(feature_name, cat_columns):
    return feature_name if '_'.join(feature_name.split('_')[:-1]) not in cat_columns else '_'.join(
        feature_name.split("_")[:-1])


def traverse_tree_path(example, tree, feature_names):
    left_child = tree.children_left
    right_child = tree.children_right
    node = 0  # start at the root
    path = []
    path_nodes = [node]

    while left_child[node] != right_child[node]:  # while not a leaf node
        feature = tree.feature[node]
        threshold = tree.threshold[node]

        # Determine which child to go based on feature threshold
        if example[feature] <= threshold:
            decision = f"Feature {feature} ({feature_names[feature]}) <= {threshold:.2f} -> LEFT"
            path.append(decision)
            node = left_child[node]
        else:
            decision = f"Feature {feature} ({feature_names[feature]}) > {threshold:.2f} -> RIGHT"
            path.append(decision)
            node = right_child[node]

        path_nodes.append(node)

    # Print the decisions taken
    for step, decision in enumerate(path, 1):
        print(f"Step {step}: {decision}")

    return path_nodes


def feature_ranges_for_path(tree, path):
    # Initialize ranges for each feature to (-inf, inf)
    feature_ranges = [(float('-inf'), float('inf')) for _ in range(tree.n_features)]

    for i, node in enumerate(path[:-1]):  # Exclude the last node to avoid out-of-range index
        feature = tree.feature[node]
        threshold = tree.threshold[node]
        # Check if this is a terminal node
        if feature == -2:
            continue
        # If the next node in the path is the left child, update upper bound
        if path[i + 1] == tree.children_left[node]:
            feature_ranges[feature] = (feature_ranges[feature][0], threshold)
        # If the next node in the path is the right child, update lower bound
        else:
            feature_ranges[feature] = (threshold, feature_ranges[feature][1])

    return feature_ranges


def generate_categorical_questions(orig_feature_name, feature_value):
    # Ask about the feature value being different
    truthiness = feature_value == 0
    return [
        {
            'q': f"If {orig_feature_name} had been different, would the outcome have been the same?",
            'a': False,
        }
    ]


def generate_continuous_questions(orig_feature_name, feature_value, min_val, max_val):
    # If the feature value is closer to the lower bound, ask about it being larger
    questions = []
    if feature_value < max_val:
        gap = max_val - feature_value
        truthiness = max_val == float('inf')
        questions.append(
            {
                'q': f"If {orig_feature_name} had been significantly larger than {feature_value:.2f}, would the outcome have been the same?",
                'a': truthiness,
            }
        )

    # If the feature value is closer to the upper bound, ask about it being smaller
    if feature_value > min_val:
        gap = feature_value - min_val
        truthiness = min_val == float('-inf')
        questions.append(
            {
                'q': f"If {orig_feature_name} had been significantly smaller than {feature_value:.2f}, would the outcome have been the same?",
                'a': truthiness,
            }
        )
    return questions


def _generate_questions(feature_values, feature_ranges, feature_names, cat_columns, orig_feature_names):
    questions = []
    considered_feature_names = []
    for i, (min_val, max_val) in enumerate(feature_ranges):
        orig_feature_name = original_feature_name(feature_names[i], cat_columns)
        is_cat = orig_feature_name in cat_columns
        # If the feature value is right on the boundary, we might not want to pose a question.
        if feature_values[i] == min_val or feature_values[i] == max_val:
            continue

        # skip the unbounded features (for now)
        if min_val == float('-inf') and max_val == float('inf'):
            continue

        considered_feature_names.append(orig_feature_name)

        qset = []
        if is_cat:
            qset = generate_categorical_questions(orig_feature_name, feature_values[i])
        else:
            qset = generate_continuous_questions(orig_feature_name, feature_values[i], min_val, max_val)
        questions.extend(qset)
    # add question about whether a random continuous feature was considered
    cont_feat = random.choice([f for f in orig_feature_names if f not in cat_columns])
    cat_feat = random.choice(cat_columns)
    for feat in [cont_feat, cat_feat]:
        questions.append(
            {
                'q': f"Was {feat} considered?",
                'a': feat in considered_feature_names,
            }
        )
    return questions


def generate_questions(example, tree, feature_names, cat_columns, orig_feature_names):
    path_nodes = traverse_tree_path(example, tree, feature_names)
    ranges = feature_ranges_for_path(tree, path_nodes)
    feature_values = example.values
    questions = _generate_questions(feature_values, ranges, feature_names, cat_columns, orig_feature_names)
    return questions