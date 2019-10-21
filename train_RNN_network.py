import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np


file_path = 'data/input/dermatology.csv'
data = pd.read_csv(file_path)


data = data.dropna(axis=0)
features = ['erythema','scaling','definite_borders','itching','koebner_phenomenon','polygonal_papules','follicular_papules',
            'oral_mucosal_involvement','knee_and_elbow_involvement','scalp_involvement','family_history',
            'melanin_incontinence','eosinophils_in_the_infiltrate','pnl_infiltrate','fibrosis_of_the_papillary_dermis','exocytosis',
            'acanthosis','hyperkeratosis','parakeratosis','clubbing_of_the_rete_ridges','elongation_of_the_rete_ridges',
            'thinning_of_the_suprapapillary_epidermis','spongiform_pustule','munro_microabcess','focal_hypergranulosis',
            'disappearance_of_the_granular_layer','vacuolisation_and_damage_of_basal_layer','spongiosis',
            'saw-tooth_appearance_of_retes','follicular_horn_plug','perifollicular_parakeratosis','inflammatory_monoluclear_inflitrate',
            'band-like_infiltrate','age']


X = data[features]
y = data.classe


model = DecisionTreeRegressor(random_state=1)

model.fit(X, y)

# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# print(model.predict(X.head()))


predicted_vals = model.predict(X)
mean_absolute_error(y, predicted_vals)


train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=25, random_state = 42)
# Define model
# model = DecisionTreeRegressor()
# Fit model
# model.fit(train_X, train_y)

# get predicted prices on validation data

# val_predictions = model.predict(val_X)
# print(mean_absolute_error(val_y, val_predictions))



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=42)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# for max_leaf_nodes in [5,10,50,150]:
#     mae = get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
#     print("mae: "+str(mae))

# print(model.predict([X.iloc[0]]))

# file_path_test = 'data/input/dermatology-test.csv'
# data_test = pd.read_csv(file_path_test)
# # data_test = data_test.dropna(axis=0)
#
#
# X_test = data[features]
#
#
#
# print("prediction "+ str(model.predict([X_test.iloc[34]])))
# print (X.iloc[0])




#################### RandomForestRegressor



from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 100,max_depth=15, random_state = 1532)
# Train the model on training data
rf.fit(train_X, train_y)

predictions = rf.predict(val_X)
errors = abs(predictions - val_y)

mape = 100 * (errors / val_y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

import pickle


with open('RNN-model.model', 'wb') as f:
    pickle.dump(rf, f)



X_test = data[features]

# d = args["data"]

# print("prediction "+ str(rf.predict([d])))

# print (X_test.iloc[6])

