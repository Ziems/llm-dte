import pandas as pd
import sklearn.preprocessing
from main import generate_rule_based_decision_tree_explanation, generate_llm_based_decision_tree_explanation

dataset = pd.read_csv('data/mushrooms.csv')

X = dataset.drop(['class'],axis=1)
y = dataset['class']
X = pd.get_dummies(X)
y = y.replace(['e', 'p'], ['edible', 'poisonous'])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

clf = sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
clf.fit(X_train, y_train)


orig_feature_names = dataset.columns.values

cat_columns = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']

feature_desc_map = {
    'cap-shape': 'the shape of the mushroom cap (e.g., bell, conical, convex, flat, knobbed, sunken)',
    'cap-surface': 'the texture of the mushroom cap surface (e.g., fibrous, grooves, scaly, smooth)',
    'cap-color': 'the color of the mushroom cap (e.g., brown, pink, white, red, buff, purple, cinnamon, green)',
    'bruises': 'presence of bruises on the mushroom (e.g., bruises or no bruises)',
    'odor': 'the smell of the mushroom (e.g., almond, anise, creosote, fishy, foul, musty, none, pungent, spicy)',
    'gill-attachment': 'how the gills are attached to the stem (e.g., attached, descending, free, notched)',
    'gill-spacing': 'spacing between the gills (e.g., close, crowded, distant)',
    'gill-size': 'the size of the gills (e.g., broad, narrow)',
    'gill-color': 'the color of the gills (e.g., black, brown, buff, chocolate, gray, green, orange, pink, purple, red, white, yellow)',
    'stalk-shape': 'the shape of the stalk (e.g., enlarging, tapering)',
    'stalk-root': 'the root of the stalk (e.g., bulbous, club, cup, equal, rhizomorphs, rooted, missing)',
    'stalk-surface-above-ring': 'the texture of the stalk surface above the ring (e.g., fibrous, scaly, silky, smooth)',
    'stalk-surface-below-ring': 'the texture of the stalk surface below the ring (e.g., fibrous, scaly, silky, smooth)',
    'stalk-color-above-ring': 'the color of the stalk above the ring (e.g., brown, buff, cinnamon, gray, orange, pink, red, white, yellow)',
    'stalk-color-below-ring': 'the color of the stalk below the ring (e.g., brown, buff, cinnamon, gray, orange, pink, red, white, yellow)',
    'veil-type': 'the type of veil (e.g., partial, universal)',
    'veil-color': 'the color of the veil (e.g., brown, orange, white, yellow)',
    'ring-number': 'the number of rings on the stalk (e.g., none, one, two)',
    'ring-type': 'the type of ring (e.g., cobwebby, evanescent, flaring, large, none, pendant, sheathing, zone)',
    'spore-print-color': 'the color of the mushroom spore print (e.g., black, brown, buff, chocolate, green, orange, purple, white, yellow)',
    'population': 'the population distribution of the mushroom (e.g., abundant, clustered, numerous, scattered, several, solitary)',
    'habitat': 'the habitat where the mushroom grows (e.g., grasses, leaves, meadows, paths, urban, waste, woods)'
}

tree_data = {
    'test_features': X_test,
    'test_labels': y_test,
    'classifier': clf,
    'feature_names': orig_feature_names,
    'categorical_columns': cat_columns,
    'original_data': dataset,
    'feature_description': feature_desc_map,
    'data_index': 5
}

generate_rule_based_decision_tree_explanation(tree_data)
generate_llm_based_decision_tree_explanation(tree_data)
