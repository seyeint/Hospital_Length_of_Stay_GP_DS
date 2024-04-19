import sklearn

import graphviz
from sklearn import ensemble
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
import logging
import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from gplearn_MLAA.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from keras.constraints import maxnorm


def main():
    # load data
    #for seed in range(0,10): we are not using this now to test arguments on ML models

    seed = 15
    df = pd.read_csv('/Users/joseferreira/Desktop/LengthOfStay.csv')

    # small preprocessing needed
    df['gender'] = df['gender'].map({'F': 1, 'M': 0})
    df['rcount'] = df['rcount'].replace('5+', 8)

    # extra special case preprocessing
    df = df[df.rcount != 8]

    # getting our needed features b4 modelling
    feature_names = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma',
                     'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor',
                     'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo',
                     'hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro',
                     'creatinine', 'bmi', 'pulse', 'respiration',
                     'secondarydiagnosisnonicd9', 'facid']

    continuous_numerical = ['rcount',
                            'hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro',
                            'creatinine', 'bmi', 'pulse', 'respiration',
                            'secondarydiagnosisnonicd9']

    # scaling
    df[continuous_numerical] = StandardScaler().fit_transform(df[continuous_numerical])

    target_name = 'lengthofstay'

    X = df[feature_names]
    y = df[target_name]

    # manually pick categorical variables
    categorical_names = ['facid']
    X.loc[:, categorical_names] = X[categorical_names].astype('object')
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.30, random_state=seed)

    # print baseline
    print("The baseline MAE: {:.3f}".format(mean_absolute_error(y_test, np.repeat(y_test.mean(), len(y_test)))))
    print("The baseline MSE: {:.3f}".format(mean_squared_error(y_test, np.repeat(y_test.mean(), len(y_test)))))
    base_mse = mean_squared_error(y_test, np.repeat(y_test.mean(), len(y_test)))

    # setup logger
    name = "_main_"
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "logFiles/" + str(datetime.datetime.now().date()) + name + "log.txt")
    logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')

    edda_params = {"deme_size": 50, "p_gsgp_demes": 0.50, "maturation": 5, "p_mutation": 1.0, "gsm_ms": -1.0}

    # models
    scores = []
    models = []

    #GP------------------------------------------------------------------------------------------------------------------------

    sr0 = SymbolicRegressor(population_size=250, init_method='grow',
                            generations=15, edda_params=None, stopping_criteria=0.0, tie_stopping_criteria=0.0,
                            edv_stopping_criteria=0.0, n_semantic_neighbors=10,
                            p_crossover=0.1, p_subtree_mutation=0.9,
                            p_gs_crossover=0.0, p_gs_mutation=0.0, gsm_ms=-1, semantical_computation=False,
                            p_hoist_mutation=0.0, p_point_mutation=0.0,
                            parsimony_coefficient=0.0, val_set=0.2,
                            verbose=1, n_jobs=1, log=True, random_state=seed)

    sr1 = SymbolicRegressor(population_size=250, init_method='grow',
                            generations=15, edda_params=None, stopping_criteria=0.0, tie_stopping_criteria=0.0,
                            edv_stopping_criteria=0.0, n_semantic_neighbors=10,
                            p_crossover=0.1, p_subtree_mutation=0.9,
                            p_gs_crossover=0.0, p_gs_mutation=0.0, gsm_ms=-1, semantical_computation=False,
                            p_hoist_mutation=0.0, p_point_mutation=0.0,
                            parsimony_coefficient=0.0, val_set=0.2,
                            verbose=1, n_jobs=1, log=True, random_state=seed)

    sr2 = SymbolicRegressor(population_size=250, init_method='grow',
                            generations=15, edda_params=None, stopping_criteria=0.0, tie_stopping_criteria=0.0,
                            edv_stopping_criteria=0.15, n_semantic_neighbors=10,
                            p_crossover=0.1, p_subtree_mutation=0.9,
                            p_gs_crossover=0.0, p_gs_mutation=0.0, gsm_ms=-1, semantical_computation=False,
                            p_hoist_mutation=0.0, p_point_mutation=0.0,
                            parsimony_coefficient=0.0, val_set=0.2,
                            verbose=1, n_jobs=1, log=True, random_state=seed)

    sr3 = SymbolicRegressor(population_size=250, init_method='',
                            generations=15, edda_params=None, stopping_criteria=0.0, tie_stopping_criteria=0.0,
                            edv_stopping_criteria=0.30, n_semantic_neighbors=10,
                            p_crossover=0.1, p_subtree_mutation=0.9,
                            p_gs_crossover=0.0, p_gs_mutation=0.0, gsm_ms=-1, semantical_computation=True,
                            p_hoist_mutation=0.0, p_point_mutation=0.0,
                            parsimony_coefficient=0.0, val_set=0.2,
                            verbose=1, n_jobs=1, log=True, random_state=seed)

    sr4 = SymbolicRegressor(population_size=250, init_method='grow',
                            generations=15, edda_params=None, stopping_criteria=0.0, tie_stopping_criteria=0.0,
                            edv_stopping_criteria=0.45, n_semantic_neighbors=10,
                            p_crossover=0.1, p_subtree_mutation=0.9,
                            p_gs_crossover=0.0, p_gs_mutation=0.0, gsm_ms=-1, semantical_computation=False,
                            p_hoist_mutation=0.0, p_point_mutation=0.0,
                            parsimony_coefficient=0.0, val_set=0.2,
                            verbose=1, n_jobs=1, log=True, random_state=seed)

    sr5 = SymbolicRegressor(population_size=250, init_method='grow',
                            generations=15, edda_params=None, stopping_criteria=0.0, tie_stopping_criteria=0.0,
                            edv_stopping_criteria=0.0, n_semantic_neighbors=10,
                            p_crossover=0.0, p_subtree_mutation=0.0,
                            p_gs_crossover=0.1, p_gs_mutation=0.9, gsm_ms=-1, semantical_computation=False,
                            p_hoist_mutation=0.0, p_point_mutation=0.0,
                            parsimony_coefficient=0.0, val_set=0.2,
                            verbose=1, n_jobs=1, log=True, random_state=seed)

    sr6 = SymbolicRegressor(population_size=500,
                            generations=30, edda_params=edda_params, stopping_criteria=0.0, tie_stopping_criteria=0.0,
                            edv_stopping_criteria=0.0, n_semantic_neighbors=10,
                            p_crossover=0.0, p_subtree_mutation=0.0,
                            p_gs_crossover=0.2, p_gs_mutation=0.8, gsm_ms=-1, semantical_computation=False,
                            p_hoist_mutation=0.0, p_point_mutation=0.0, val_set=0.2,
                            verbose=1, n_jobs=1, log=True, random_state=seed)

    sr7 = SymbolicRegressor(population_size=250,
                            generations=15, edda_params=edda_params, stopping_criteria=0.0, tie_stopping_criteria=0.15,
                            edv_stopping_criteria=0.0, n_semantic_neighbors=10,
                            p_crossover=0.0, p_subtree_mutation=0.0,
                            p_gs_crossover=0.1, p_gs_mutation=0.9, gsm_ms=-1, semantical_computation=False,
                            p_hoist_mutation=0.0, p_point_mutation=0.0, val_set=0.2,
                            verbose=1, n_jobs=1, log=True, random_state=seed)

    sr8 = SymbolicRegressor(population_size=250,
                            generations=15, edda_params=None, stopping_criteria=0.0, tie_stopping_criteria=0.0,
                            edv_stopping_criteria=0.30, n_semantic_neighbors=10,
                            p_crossover=0.0, p_subtree_mutation=0.0,
                            p_gs_crossover=0.1, p_gs_mutation=0.9, gsm_ms=-1, semantical_computation=False,
                            p_hoist_mutation=0.0, p_point_mutation=0.0,
                            parsimony_coefficient=0.0, val_set=0.2,
                            verbose=1, n_jobs=1, log=True, random_state=seed)

    sr9 = SymbolicRegressor(population_size=250,
                            generations=15, edda_params=None, stopping_criteria=0.0, tie_stopping_criteria=0.0,
                            edv_stopping_criteria=0.45, n_semantic_neighbors=10,
                            p_crossover=0.0, p_subtree_mutation=0.0,
                            p_gs_crossover=0.1, p_gs_mutation=0.9, gsm_ms=-1, semantical_computation=False,
                            p_hoist_mutation=0.0, p_point_mutation=0.0,
                            parsimony_coefficient=0.0, val_set=0.2,
                            verbose=1, n_jobs=1, log=True, random_state=seed)
    #NN------------------------------------------------------------------------------------------------------------------------
    input_dim = X.shape[1]
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    #model.summary()

    model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae'])
    #history = model.fit(X_train, y_train, epochs=150, batch_size=50, verbose=1, validation_split=0.2)

    #plot
    # print(history.history.keys())
    # # "Loss"
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    #Random Forest------------------------------------------------------------------------------------------------------------------------
    #rf = RandomForestRegressor(random_state=seed)
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=15)]
    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    #max_depth.append(None)
    # Minimum number of samples required to split a node
    #min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    #min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    #bootstrap = [True, False]
    # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    #
    # rf = RandomForestRegressor()
    # rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=2, verbose=2,
    #                                random_state=seed, n_jobs=-1)
    # rf_random.fit(X_train, y_train)
    #
    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))

        return accuracy

    #best_random = rf_random.best_estimator_
    #random_accuracy = evaluate(best_random, X_test, y_test)

    #Bagging------------------------------------------------------------------------------------------------------------------------
    # grid = ParameterGrid({"max_samples": [0.5, 1.0],
    #                       "max_features": [0.5, 1.0],
    #                       "bootstrap": [True, False],
    #                       "bootstrap_features": [True, False]})
    #
    # for base_estimator in [None,
    #                        DummyRegressor(),
    #                        DecisionTreeRegressor(),
    #                        KNeighborsRegressor(),
    #                        SVR()]:
    #     for params in grid:
    #         BaggingRegressor(base_estimator=base_estimator,
    #                          random_state=seed,
    #                          **params).fit(X_train, y_train).predict(X_test)

    #Gradient Boost------------------------------------------------------------------------------------------------------------------------

    # Fit regression model
    # params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2,
    #           'learning_rate': 0.01, 'loss': 'ls'}
    # model = GradientBoostingRegressor(**params)
    #
    # model.fit(X_train, y_train)
    # y_predicted = model.predict(X_test)
    # print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
    # print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_predicted))
    # print("R2 score: %.2f" % r2_score(y_test, y_predicted))

    #AdaBoost------------------------------------------------------------------------------------------------------------------------
    boost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(),
                              random_state=seed)
    parameters = {'n_estimators': (10,50),
                  'base_estimator__max_depth': (5,10)}
    clf = GridSearchCV(boost, parameters)
    clf.fit(X_train, y_train)
    ada = AdaBoostRegressor()
    search_grid = {'n_estimators': [500], 'learning_rate': [0.01, .1], 'random_state': [seed]}
    search = GridSearchCV(estimator=ada, param_grid=search_grid, scoring='neg_mean_squared_error', n_jobs=1,
                          cv=2)
    search.fit(X, y)
    random_accuracy = evaluate(search.best_estimator_, X_test, y_test)

    #Bagging------------------------------------------------------------------------------------------------------------------------
    # model2 = sklearn.ensemble.BaggingRegressor(n_estimators=1000, max_samples=1000, max_features=20)
    # model2.fit(X_train, y_train)
    #
    # evaluate(model2, X_test, y_test)




    models.extend([('sr', sr7)
                   ])

    #Evaluation
    for model in models:
        model[1].fit(X_train, y_train)
        y_test_pred = model[1].predict(X_test)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)  # no need to print it

        rsquared = 1 - (mse / base_mse)
        print('Model {}: \n mae: {} \n mse: {} \n r2: {} \n'.format(model[0], mae, mse, rsquared))
        scores.append((model[0], [mae, mse, rsquared]))


    # Visualization of a SR singular

    # Print fittest solution
    # print(sr._program)

    # Export to a graph instance
    # graph = sr._program.export_graphviz()
    # graph_str = str(graph)
    # program_str = str(sr._program)

    # Replace X{} with actual features names
    # mapping_dict = {'X{}'.format(i): X.columns[i] for i in reversed(range(X.shape[1]))}

    # for old_value, new_value in mapping_dict.items():
    #   graph_str = graph_str.replace(old_value, new_value)
    #  program_str = program_str.replace(old_value, new_value)

    # print('Readable:',program_str)

    # Amazing Tree (or not)!
    # src = graphviz.Source(graph_str)
    # src.render('result.gv', view=True)

    print('FINALLY HERE IS THE JUICE:', scores)


if __name__ == "__main__":
    main()