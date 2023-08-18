# Databricks notebook source
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, SparkTrials, Trials, fmin, hp, tpe
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, min, when
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

logging.basicConfig(
    format="%(asctime)s %(filename)s %(funcName)s %(lineno)d %(message)s"
)
logger = logging.getLogger("Machine Learning Techniques")
logger.setLevel(logging.INFO)

MODELS_PARAMS = {
    "lightGBM": {
        "fixed_params": {
            "boosting_type": "gbdt",
            "objective": "binary",
            "bagging_freq": 5,
            "bagging_fraction": 0.4,
            "min_split_gain": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "min_data_in_leaf": 20,
            "subsample_freq": 0,
            "subsample_for_bin": 20,
            "min_split_gain": 0,
            "reg_alpha": 0,
            "reg_lambda": 0,
            "nthread": 4,
            "verbose": 0,
            "early_stopping_rounds": 20,
            "n_jobs": -1,
        },
        "variable_params": {
            "max_depth": hp.choice("max_depth", np.arange(5, 15, dtype=int)),
            "num_leaves": hp.quniform("num_leaves", 100, 200, 20),
            "min_child_samples": hp.quniform("min_child_samples", 30, 180, 30),
            "pos_bagging_fraction": hp.quniform(
                "pos_bagging_fraction", 0.05, 0.95, 0.1
            ),
            "neg_bagging_fraction": hp.quniform(
                "neg_bagging_fraction", 0.05, 0.95, 0.1
            ),
            "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
            "max_bin": hp.choice("max_bin", [50, 75, 100, 125]),
            "colsample_bytree": hp.quniform("colsample_bytree", 0.2, 1, 0.1),
            "importance_type": hp.choice("importance_type", ["split", "gain"]),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
            "subsample": hp.quniform("subsample", 0.4, 1, 0.1),
        },
    },
    "XGBoost": {
        "fixed_params": {
            "booster": "gbtree",
            "gamma": 0.0,
            "max_delta_step": 0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "n_jobs": -1,
        },
        "variable_params": {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
            "max_depth": hp.choice("max_depth", np.arange(1, 10, dtype=int)),
            "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        },
    },
    "randomForest": {
        "fixed_params": {
            "min_weight_fraction_leaf": 0.0,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": -1,
            "class_weight": "balanced",
            "warm_start": False,
        },
        "variable_params": {
            "n_estimators": hp.quniform("n_estimators", 100, 1000, 100),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "max_depth": hp.choice("max_depth", np.arange(5, 25, dtype=int)),
            "min_samples_split": hp.choice(
                "min_samples_split", np.arange(2, 10, dtype=int)
            ),
            "min_samples_leaf": hp.choice(
                "min_samples_leaf", np.arange(1, 10, dtype=int)
            ),
            "max_features": hp.choice("max_features", [None, "auto", "sqrt", "log2"]),
        },
    },
}


class MLTechniques:
    """Machine Learning techniques for classification (XGBoost, LightGBM, Random Forest)"""

    def xgb_teq(param, x_train, y_train, x_test, y_test, n_splits=5):
        """
        Performs K-fold CV over a given training set and test set for XGBoost
        and outputs log loss metrics and prediction/true label outputs.
        Args:
            x_train (np.array): Full training set.
            y_train (np.array): Full training set labels.
            x_test (np.array): Full test set.
            y_test (np.array): Full test set labels.
            n_splits (int): Number of K-fold splits for K-fold CV.

        Returns:
            ts_xgb (list): List of log loss scores from each training set k fold split.
            cvs_xgb (list): List of log loss scores from each validation set k fold split.
            log_loss_xgboost_gradient_boosting (float): Entire training set log loss score.
            predictions_test_set_xgb (pd.DataFrame): Single-column dataframe of test set predictions.
            preds (pd.DataFrame): Two-column dataframe of test set true labels and predictions.
            gbdt (object): XGBoost train object.
        """

        # Read in best parameters for the model
        params_xgb = {
            "booster": param["booster"],
            "colsample_bytree": param["colsample_bytree"],
            "gamma": param["gamma"],
            "learning_rate": param["learning_rate"],
            "max_delta_step": param["max_delta_step"],
            "max_depth": param["max_depth"],
            "min_child_weight": param["min_child_weight"],
            "n_jobs": param["n_jobs"],
            "objective": param["objective"],
            "reg_alpha": param["reg_alpha"],
            "reg_lambda": param["reg_lambda"],
            "subsample": param["subsample"],
        }

        # Make a data frame of the size of y_train with one column - 'prediction'. Then make a Stratified
        # K fold object with n_splits
        ts_xgb = []
        cvs_xgb = []
        predictions_based_on_kfolds = pd.DataFrame(
            data=[], index=y_train.index, columns=["prediction"]
        )
        k_fold = StratifiedKFold(n_splits, shuffle=True)

        for train_index, cv_index in k_fold.split(
            np.zeros(len(x_train)), y_train.ravel()
        ):

            # Take subsets of X_train and y_train based on the K-fold splits
            X_train_fold, X_cv_fold = (
                x_train.iloc[train_index, :],
                x_train.iloc[cv_index, :],
            )
            y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

            # Convert those splits into DMatrix datasets, which are designed to interface with xgb models
            dtrain = xgb.DMatrix(data=X_train_fold, label=y_train_fold)
            dCV = xgb.DMatrix(data=X_cv_fold)

            # Use best params and the dtrain dataset to perform K fold cross validation on the (already k-folded) dataset.
            res = xgb.cv(
                params_xgb,
                dtrain,
                num_boost_round=50,
                nfold=5,
                early_stopping_rounds=10,
                verbose_eval=5,
            )
            best_nrounds = res.shape[0] - 1
            print(
                np.shape(x_train), np.shape(x_test), np.shape(y_train), np.shape(y_test)
            )

            # Train on dtrain
            gbdt = xgb.train(params_xgb, dtrain, best_nrounds)

            # Calculate training log_loss between real labels and predicted labels for this fold. Then store
            # this value in a list so that we can later take the average of all of the single fold log loss values
            log_loss_training = log_loss(y_train_fold, gbdt.predict(dtrain))
            ts_xgb.append(log_loss_training)

            # Predict on the val set and insert the predictions from this fold into the dataframe created outside the loop
            predictions_based_on_kfolds.loc[
                X_cv_fold.index, "prediction"
            ] = gbdt.predict(dCV)

            # Calculate val set log_loss between real labels and predicted labels for this fold. Then store
            # this value in a list so that we can later take the average of all of the single fold log loss values
            log_loss_cv = log_loss(
                y_cv_fold,
                predictions_based_on_kfolds.loc[X_cv_fold.index, "prediction"],
            )
            cvs_xgb.append(log_loss_cv)

        # Now that we have looped through all folds, calculate the total log loss across all data
        log_loss_xgboost_gradient_boosting = log_loss(
            y_train, predictions_based_on_kfolds.loc[:, "prediction"]
        )

        # Join up the training set labels with the predicted labels
        preds = pd.concat(
            [y_train, predictions_based_on_kfolds.loc[:, "prediction"]], axis=1
        )
        preds.columns = ["trueLabel", "prediction"]

        # Make a dmatrix out of the test set, and predict on it, then store it in a dataframe.
        dtr = xgb.DMatrix(data=x_test, label=y_test)
        predictions_test_set_xgb = pd.DataFrame(
            data=[], index=y_test.index, columns=["prediction"]
        )
        predictions_test_set_xgb.loc[:, "prediction"] = gbdt.predict(dtr)

        return (
            ts_xgb,
            cvs_xgb,
            log_loss_xgboost_gradient_boosting,
            predictions_test_set_xgb,
            preds,
            gbdt,
        )

    def lgbm_teq(
        param, x_train, y_train, x_test, y_test, scale_pos_weight=1, n_splits=5
    ):
        """
        Performs K-fold CV over a given training set and test set for LightGBM
        and outputs log loss metrics and prediction/true label outputs.

        Args:
            x_train (np.array): Full training set.
            y_train (np.array): Full training set labels.
            x_test (np.array): Full test set.
            y_test (np.array): Full test set labels.
            scale_pos_weight (int): assumeing balanced dataset at 1
            n_splits (int): Number of K-fold splits for K-fold CV.

        Returns:
            ts_lightgbm (list): List of log loss scores from each training set k fold split.
            cvs_lightgbm (list): List of log loss scores from each validation set k fold split.
            log_loss_xgboost_gradient_boosting (float): Entire training set log loss score.
            predictions_test_set_xgb (pd.DataFrame): Single-column dataframe of test set predictions.
            preds (pd.DataFrame): Two-column dataframe of test set true labels and predictions.
            gbm (object): LightGBM train object.
            bestiteration (int): The best LightGBM iteration.
        """

        # Read in best parameters for the model
        params_lightGB = {
            "bagging_fraction": param["bagging_fraction"],
            "objective": param["objective"],
            "bagging_freq": param["bagging_freq"],
            "boosting_type": param["boosting_type"],
            "colsample_bytree": param["colsample_bytree"],
            "neg_bagging_fraction": param["neg_bagging_fraction"],
            "pos_bagging_fraction": param["pos_bagging_fraction"],
            "importance_type": param["importance_type"],
            "learning_rate": param["learning_rate"],
            "max_depth": param["max_depth"],
            "min_child_samples": param["min_child_samples"],
            "min_data_in_leaf": param["min_data_in_leaf"],
            "max_bin": param["max_bin"],
            "min_child_weight": param["min_child_weight"],
            "subsample_for_bin": param["subsample_for_bin"],
            "nthread": param["nthread"],
            "verbose": param["verbose"],
            "early_stopping_rounds": param["early_stopping_rounds"],
            "min_split_gain": param["min_split_gain"],
            "n_jobs": param["n_jobs"],
            "num_leaves": param["num_leaves"],
            "subsample_freq": param["subsample_freq"],
            "reg_alpha": param["reg_alpha"],
            "reg_lambda": param["reg_lambda"],
            "scale_pos_weight": scale_pos_weight,
            "subsample": param["subsample"],
        }

        # Make a data frame of the size of y_train with one column - 'prediction'. Then make a Stratified
        # K fold object with n_splits
        ts_lightgbm = []
        cvs_lightgbm = []
        predictions_based_on_kfolds = pd.DataFrame(
            data=[], index=y_train.index, columns=["prediction"]
        )
        k_fold = StratifiedKFold(n_splits, shuffle=True)

        for train_index, cv_index in k_fold.split(
            np.zeros(len(x_train)), y_train.ravel()
        ):

            # Take subsets of X_train and y_train based on the K-fold splits
            X_train_fold, X_cv_fold = (
                x_train.iloc[train_index, :],
                x_train.iloc[cv_index, :],
            )
            y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

            # Convert those splits into DMatrix datasets, which are designed to interface with lgbm models, then train
            lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
            lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference=lgb_train)
            gbm = lgb.train(
                params_lightGB,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                early_stopping_rounds=200,
            )

            # Calculate training log_loss between real labels and predicted labels for this fold. Then store
            # this value in a list so that we can later take the average of all of the single fold log loss values
            log_loss_training = log_loss(
                y_train_fold,
                gbm.predict(X_train_fold, num_iteration=gbm.best_iteration),
            )
            ts_lightgbm.append(log_loss_training)

            # Predict on the val set and insert the predictions from this fold into the dataframe created outside the loop
            predictions_based_on_kfolds.loc[
                X_cv_fold.index, "prediction"
            ] = gbm.predict(X_cv_fold, num_iteration=gbm.best_iteration)

            # Calculate val set log_loss between real labels and predicted labels for this fold. Then store
            # this value in a list so that we can later take the average of all of the single fold log loss values
            log_loss_cv = log_loss(
                y_cv_fold,
                predictions_based_on_kfolds.loc[X_cv_fold.index, "prediction"],
            )
            cvs_lightgbm.append(log_loss_cv)

        # Now that we have looped through all folds, calculate the total log loss across all data
        log_loss_lightgbm_gradient_boosting = log_loss(
            y_train, predictions_based_on_kfolds.loc[:, "prediction"]
        )

        # Join up the training set labels with the predicted labels
        preds = pd.concat(
            [y_train, predictions_based_on_kfolds.loc[:, "prediction"]], axis=1
        )
        preds.columns = ["trueLabel", "prediction"]

        # Prepare outputs
        predictions_test_set_lightgbm = pd.DataFrame(
            data=[], index=y_test.index, columns=["prediction"]
        )
        predictions_test_set_lightgbm.loc[:, "prediction"] = gbm.predict(
            x_test, num_iteration=gbm.best_iteration
        )
        log_loss_test_set_lightgbm = log_loss(y_test, predictions_test_set_lightgbm)
        bestiteration = gbm.best_iteration

        return (
            ts_lightgbm,
            cvs_lightgbm,
            log_loss_lightgbm_gradient_boosting,
            predictions_test_set_lightgbm,
            preds,
            gbm,
            bestiteration,
        )

    def random_forest_teq(param, x_train, y_train, x_test, y_test, n_splits=5):
        """
        Performs K-fold CV over a given training set and test set for Random Forest
        and outputs log loss metrics and prediction/true label outputs.

        Args:
            param (dict): Dictionary of hyperparameters for Random Forest.
            x_train (np.array): Full training set.
            y_train (np.array): Full training set labels.
            x_test (np.array): Full test set.
            y_test (np.array): Full test set labels.
            n_splits (int): Number of K-fold splits for K-fold CV.

        Returns:
            ts_rf (list): List of log loss scores from each training set k-fold split.
            cvs_rf (list): List of log loss scores from each validation set k-fold split.
            log_loss_random_forest (float): Entire training set log loss score.
            predictions_test_set_rf (pd.DataFrame): DataFrame of test set predictions.
            preds (pd.DataFrame): DataFrame of test set true labels and predictions.
            rf_model (object): Fitted Random Forest model.
            best_estimators (int): Number of estimators of the best Random Forest model.
        """

        # Read in the parameters for the Random Forest model
        params_rf = {
            "n_estimators": int(param["n_estimators"]),
            "max_depth": int(param["max_depth"]),
            "min_samples_split": int(param["min_samples_split"]),
            "min_samples_leaf": int(param["min_samples_leaf"]),
            "max_features": param["max_features"],
            # 'random_state': param['random_state']
        }

        # Make a data frame of the size of y_train with one column - 'prediction'.
        # Then make a Stratified K fold object with n_splits
        ts_rf = []
        cvs_rf = []
        predictions_based_on_kfolds = pd.DataFrame(
            data=[], index=y_train.index, columns=["prediction"]
        )
        k_fold = StratifiedKFold(n_splits, shuffle=True)

        for train_index, cv_index in k_fold.split(
            np.zeros(len(x_train)), y_train.ravel()
        ):
            # Take subsets of X_train and y_train based on the K-fold splits
            X_train_fold, X_cv_fold = (
                x_train.iloc[train_index, :],
                x_train.iloc[cv_index, :],
            )
            y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

            # Create and fit the Random Forest model
            rf = RandomForestClassifier(**params_rf)
            rf.fit(X_train_fold, y_train_fold)

            # Calculate training log loss between real labels and predicted labels for this fold
            log_loss_training = log_loss(y_train_fold, rf.predict_proba(X_train_fold))
            ts_rf.append(log_loss_training)

            # Predict on the validation set and insert the predictions from this fold into the dataframe
            predictions_based_on_kfolds.loc[
                X_cv_fold.index, "prediction"
            ] = rf.predict_proba(X_cv_fold)[:, 1]

            # Calculate validation set log loss between real labels and predicted labels for this fold
            log_loss_cv = log_loss(
                y_cv_fold,
                predictions_based_on_kfolds.loc[X_cv_fold.index, "prediction"],
            )
            cvs_rf.append(log_loss_cv)

        # Calculate the total log loss across all data
        log_loss_rf = log_loss(
            y_train, predictions_based_on_kfolds.loc[:, "prediction"]
        )

        # Join up the training set labels with the predicted labels
        preds = pd.concat(
            [y_train, predictions_based_on_kfolds.loc[:, "prediction"]], axis=1
        )
        preds.columns = ["trueLabel", "prediction"]

        # Prepare outputs
        predictions_test_set_rf = pd.DataFrame(
            data=[], index=y_test.index, columns=["prediction"]
        )
        predictions_test_set_rf.loc[:, "prediction"] = rf.predict_proba(x_test)[:, 1]
        log_loss_test_set_rf = log_loss(y_test, predictions_test_set_rf)
        best_num_estimators = rf.n_estimators

        return (
            ts_rf,
            cvs_rf,
            log_loss_rf,
            predictions_test_set_rf,
            preds,
            rf,
            best_num_estimators,
        )


# COMMAND ----------


class HPOpt(object):
    """
    The full Hyperopt class. Defines variable and fixed parameter spaces, then with an input model and metric, trains and tests while varying over the variable parameter space. Progress is tracked using the supplied metric.

    Args:
        x_train (np.array): Train-split subset of df_features as a numpy array (size determined by 1 - test_size).
        y_train (np.array): Train-split subset of df_labels as a numpy array (size determined by 1 - test_size).
        model_name (object): The sklearn (or other) classifier model to be used.
        metric_name (object): The sklearn (or other) metric to be used to perform evaluation of the model during hyperparameter tuning.
        do_cv (bool): Boolean variable to do K-fold cross-validation or not.
    """

    def __init__(
        self, x_train, y_train, model_name, metric_name, do_cv, scale_pos_weight
    ):
        assert model_name in [
            "lightGBM",
            "XGBoost",
            "randomForest",
        ], "Incorrect model chosen"
        assert metric_name in [
            "logLoss",
            "accuracyScore",
            "precisionScore",
            "recallScore",
            "f1Score",
            "ROCAUCScore",
            "prAUCScore",
        ], "Incorrect metric chosen"

        self.x_train = x_train
        self.y_train = y_train
        self.model_name = model_name
        self.metric_name = metric_name
        self.do_cv = do_cv
        self.scale_pos_weight = scale_pos_weight

        fixed_params = MODELS_PARAMS[model_name]["fixed_params"]
        variable_params = MODELS_PARAMS[model_name]["variable_params"]
        self.space = {**fixed_params, **variable_params}

    ################## Metric methods ##################
    def logLoss(self, ytrue, ypred):
        return log_loss(ytrue, ypred)

    def accuracyScore(self, ytrue, ypred):
        return accuracy_score(ytrue, ypred)

    def precisionScore(self, ytrue, ypred):
        return precision_score(ytrue, ypred)

    def recallScore(self, ytrue, ypred):
        return recall_score(ytrue, ypred)

    def f1Score(self, ytrue, ypred):
        return f1_score(ytrue, ypred)

    def ROCAUCScore(self, ytrue, ypred):
        return roc_auc_score(ytrue, ypred)

    def prAUCScore(self, ytrue, ypred):
        precision, recall, _ = precision_recall_curve(ytrue, ypred)
        return auc(recall, precision)

    ################## Model methods ##################
    def lightGBM(self, X, y, Xval, yval, params):
        """
        Train a LightGBM model on the given X, y dataset. Then predict on the training and withheld
        validation sets to get a sense of accuracy.

        Args:
            X (np.array): Feature dataset to train the model on.
            y (np.array): Label dataset to train the model on.
            Xval (np.array): Feature dataset to predict on.
            yval (np.array): Label dataset to predict on.
            params (dict): Model parameter dictionary passed in by hyperopt.
        """
        dtrain = lgb.Dataset(X, y)
        dval = lgb.Dataset(Xval, yval)
        params["scale_post_weight"] = self.scale_pos_weight
        # Train the model on the training set
        trained_model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            early_stopping_rounds=50,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
        )

        # Predict on the training and val sets - this gives training and val set results
        training_set_predictions = trained_model.predict(
            X, num_iteration=trained_model.best_iteration
        )
        val_set_predictions = trained_model.predict(
            Xval, num_iteration=trained_model.best_iteration
        )
        return training_set_predictions, val_set_predictions

    def XGBoost(self, X, y, Xval, yval, params):
        """
        Train an XGBoost model on the given X, y dataset. Then predict on the training and withheld
        validation sets to get a sense of accuracy.

        Args:
            X (np.array): Feature dataset to train the model on.
            y (np.array): Label dataset to train the model on.
            Xval (np.array): Feature dataset to predict on.
            yval (np.array): Label dataset to predict on.
            params (dict): Model parameter dictionary passed in by hyperopt.
        """
        dtrain = xgb.DMatrix(data=X, label=y)
        dval = xgb.DMatrix(data=Xval, label=yval)

        # Train the model on the training set
        trained_model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            early_stopping_rounds=50,
            evals=[(dval, "valid"), (dtrain, "train")],
        )

        # Predict on the training and val sets - this gives training and val set results
        training_set_predictions = trained_model.predict(
            xgb.DMatrix(X), ntree_limit=trained_model.best_ntree_limit
        )
        val_set_predictions = trained_model.predict(
            xgb.DMatrix(Xval), ntree_limit=trained_model.best_ntree_limit
        )
        return training_set_predictions, val_set_predictions

    def randomForest(self, X, y, Xval, yval, params):
        """
        Train a Random Forest model on the given X, y dataset. Then predict on the training and withheld
        validation sets to get a sense of accuracy.

        Args:
            X (np.array): Feature dataset to train the model on.
            y (np.array): Label dataset to train the model on.
            Xval (np.array): Feature dataset to predict on.
            yval (np.array): Label dataset to predict on.
            params (dict): Model parameter dictionary passed in by hyperopt.
        """
        # For some reason, the n_estimators parameter gets turned into a dict and everything
        # breaks if you pass it in from the params dict. So we must extract it as a variable and feed it
        # into the classifier separately. The sheerest of perversity.
        params["n_estimators"] = int(params["n_estimators"])
        clf = RandomForestClassifier(**params)

        # Train the model on the training set
        trained_model = clf.fit(X, y)

        # Predict on the training and val sets - this gives training and val set results
        training_set_predictions = trained_model.predict_proba(X)
        val_set_predictions = trained_model.predict_proba(Xval)
        return training_set_predictions[:, 1], val_set_predictions[:, 1]

    ################## Objective method ##################
    def cross_validate_with_model(self, params):
        """
        Train the supplied model on the given training data and then evaluate the supplied metric
        (accuracy, log loss, precision, etc.) on the validation set.

        Args:
            params (dict): A sampling of the parameter space (handled by hyperopt).

        Returns:
            _ (dict): A dictionary of values including the loss, parameters, and other metrics.
        """

        model_name = self.model_name
        metric_name = self.metric_name

        # This is just an annoying cleaning step. lightGBM and randomForest sometimes protest that some parameters
        # aren't integers. This just forces them to be integers.
        if model_name == "lightGBM":
            intnames = ["num_leaves"]
            for ints in intnames:
                params[ints] = int(params[ints])
        elif model_name == "randomForest":
            intnames = ["n_estimators"]
            for ints in intnames:
                params[ints] = int(params[ints])

        # The metric measured on the k folds and used as the final output. Options are the above 'Score' methods
        metric = getattr(self, metric_name)

        # If do_cv, then the metric will be measured on 5 different validation sets, and averaged. If
        # do_cv == False, it will just fit on the whole training set and measure the metric on the validation set once.
        # CV takes a lot longer but will give a more accurate assessment of the metric.
        if self.do_cv:

            k_fold_training_set_metric = []
            k_fold_val_set_metric = []
            k_fold = StratifiedKFold(5, shuffle=True)

            for train_index, cv_index in k_fold.split(
                np.zeros(len(self.x_train)), self.y_train.ravel()
            ):

                # Slice x_train and y_train up according to the indices in train_index and cv_index
                X_train_fold, X_cv_fold = (
                    self.x_train.iloc[train_index, :],
                    self.x_train.iloc[cv_index, :],
                )
                y_train_fold, y_cv_fold = (
                    self.y_train.iloc[train_index],
                    self.y_train.iloc[cv_index],
                )

                # Gets the attributes of the model method here: i.e. 'lightGBM', 'XGBoost', 'randomForest' etc.
                # and then calls that method to train.
                model = getattr(self, model_name)
                training_set_predictions, val_set_predictions = model(
                    X_train_fold, y_train_fold, X_cv_fold, y_cv_fold, params
                )

                # Most scikit metrics work on (ytrue, ypreds) where ypreds are the LABELS that have been predicted.
                # However logloss requires the PROBABILITIES of the positive class (i.e. prob of a '1') rather than 0 or 1.
                if metric_name == "logLoss":
                    training_set_metric = metric(y_train_fold, training_set_predictions)
                    val_set_metric = metric(y_cv_fold, val_set_predictions)
                else:
                    training_set_metric = metric(
                        y_train_fold, np.round(training_set_predictions).astype(float)
                    )
                    val_set_metric = metric(
                        y_cv_fold, np.round(val_set_predictions).astype(float)
                    )

                k_fold_training_set_metric.append(training_set_metric)
                k_fold_val_set_metric.append(val_set_metric)

            # Average the validation set metric results from the k folds to get an overall number
            final_training_set_metric = np.mean(k_fold_training_set_metric)
            final_val_set_metric = np.mean(k_fold_val_set_metric)

        elif self.do_cv == False:
            # If no k-fold cross-validation, then just break up into a training and validation set and measure metrics once.
            X_train_fold, X_cv_fold, y_train_fold, y_cv_fold = train_test_split(
                self.x_train, self.y_train, test_size=0.33, stratify=self.y_train
            )

            # Gets the attributes of the model method here: i.e. 'lightGBM', 'XGBoost', 'randomForest' etc.
            # and then calls that method to traißn.
            model = getattr(self, model_name)
            training_set_predictions, val_set_predictions = model(
                X_train_fold, y_train_fold, X_cv_fold, y_cv_fold, params
            )

            if metric_name == "logLoss":
                final_training_set_metric = metric(
                    y_train_fold, training_set_predictions
                )
                final_val_set_metric = metric(y_cv_fold, val_set_predictions)
            else:
                final_training_set_metric = metric(
                    y_train_fold, np.round(training_set_predictions).astype(int)
                )
                final_val_set_metric = metric(
                    y_cv_fold, np.round(val_set_predictions).astype(int)
                )

        # Hyperopt always MINIMISES whatever metric you give it. So if you give it something like accuracy (which
        # you want to be high) then you need to take the reciprocal
        if metric_name == "log_loss":
            loss = abs(final_val_set_metric)

        # f1 score, precison, recall are all [0, 1] - higher is better - so must take reciprocal
        elif metric_name in [
            "f1_score",
            "precision_score",
            "recall_score",
            "f1Score",
            "ROCAUCScore",
            "precisionScore",
            "prAUCScore",
        ]:
            loss = 1 / final_val_set_metric

        else:
            raise Exception(
                "Have you converted the loss score such that it is going to be correctly minimised by hyperopt?"
            )

        return {
            "loss": loss,
            "training-" + metric_name: final_training_set_metric,
            "val-" + metric_name: final_val_set_metric,
            "status": STATUS_OK,
            "params": params,
        }


# COMMAND ----------


# COMMAND ----------


def evaluate_trained_model(testev, y_test, threshold: float = 0.5) -> pd.DataFrame:
    """
    :info: Creates a dataframe of performance metrics for the test set. Basically a big summary of test set performance.
    :inputs:
        :pd.DataFrame testev: DataFrame of test set predictions
        :np.array y_test: yest set true labels
    :returns:
        :pd.DataFrame validation_metric: Output dataframe of validation metrics
    """

    # Set up predicted and true labels for the test set
    y_pred = testev["prediction"].values
    ytrue = y_test.values
    y_pred_binary = np.where(y_pred > threshold, 1, 0)

    # Create a dataframe of a whole bunch of validation metrics for the test set
    validation_metric = pd.DataFrame(columns=["metric", "value"])
    ROCAUCScore = roc_auc_score(ytrue, y_pred_binary)
    validation_metric = validation_metric.append(
        {"metric": "ROCAUCScore", "value": ROCAUCScore}, ignore_index=True
    )
    f1Score = f1_score(ytrue, y_pred_binary)
    validation_metric = validation_metric.append(
        {"metric": "f1Score", "value": f1Score}, ignore_index=True
    )
    precisionScore = precision_score(ytrue, y_pred_binary)
    validation_metric = validation_metric.append(
        {"metric": "precisionScore", "value": precisionScore}, ignore_index=True
    )
    accuracyScore = accuracy_score(ytrue, y_pred_binary)
    validation_metric = validation_metric.append(
        {"metric": "accuracyScore", "value": accuracyScore}, ignore_index=True
    )
    logLoss = log_loss(ytrue, y_pred)
    validation_metric = validation_metric.append(
        {"metric": "logLoss", "value": logLoss}, ignore_index=True
    )

    return validation_metric


# COMMAND ----------

import matplotlib.pyplot as plt


def plot_model_histogram(model, X_test, y_test, title: str = "Model Score Histogram"):
    """
    Plot the model scores histogram for two classes.

    Args:
        model: Trained classifier model.
        X_test: Test data.
        y_test: True labels for the test data.

    Returns:
        None
    """
    y_proba = model.predict_proba(X_test)
    y_true = y_test.to_numpy()

    probas_0 = []
    probas_1 = []

    for proba, label in zip(y_proba, y_true):
        if label == 0:
            probas_0.append(proba[1])
        else:
            probas_1.append(proba[1])

    xlim = [0.0, 1]

    fig, ax = plt.subplots(figsize=(12, 10))

    n0, bins0, patches0 = plt.hist(
        probas_0, bins=8, alpha=0.5, color="green", label="Fine", range=xlim
    )
    n1, bins1, patches1 = plt.hist(
        probas_1, bins=8, alpha=0.5, color="red", label="MBS Excluded", range=xlim
    )

    plt.xlabel("Model Score", fontsize=16)
    plt.ylabel("Customers", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.title(f"{title}", fontsize=16)
    plt.xlim(xlim)

    # Loop through the histogram bars for class 1
    for i, p in enumerate(patches1):
        # Calculate the percentage of correct predictions for the current bin
        correct_pct = n1[i] / (n0[i] + n1[i])
        # Get the x-coordinate and height of the current histogram bar
        x = p.get_x() + p.get_width() / 2
        height = p.get_height() + 1
        # Display the percentage value at the top of the current histogram bar
        plt.text(x, height, f"{correct_pct:.1%}", ha="center", va="bottom", fontsize=10)

    plt.show()


# COMMAND ----------


def split_train_test_data(
    df: pd.DataFrame,
    drop_columns: list = ["accountid", "accountnumber", "customerid", "customergroup"],
    target_column: str = "alert",
    test_size: float = 0.33,
) -> tuple:
    """
    Preprocesses the input DataFrame by dropping specified columns, splitting it into train and test sets,
    and returning the necessary data for further analysis.

    Args:
        df (pd.DataFrame): The input DataFrame.
        drop_columns (list): List of columns to drop (default: ["accountid", "accountnumber", "customerid"]).
        target_column (str): The target column (default: "alert").

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test, and df_labels.
    """
    features_df_pd_org = df.copy()

    features_df_pd = features_df_pd_org.drop(drop_columns, axis=1)

    original_features = [col for col in features_df_pd.columns if col != target_column]
    X = features_df_pd[original_features]
    y = features_df_pd[target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=99, shuffle=True
    )

    X_train = X_train.sort_index()
    y_train = y_train.sort_index()

    X_test = X_test.sort_index()
    y_test = y_test.sort_index()

    df_labels = features_df_pd[target_column].copy()

    return X_train, X_test, y_train, y_test, df_labels


def perform_parameter_tuning(df_features: pd.DataFrame, df_labels: pd.DataFrame, metric_name: object,                              
                             max_eval_param_tuning: int, explore_dir_params: int, model_names: list = ['randomForest'], parallel_tuning: bool = True) -> Dict[str, Any]:     """
    Sets up the Hyperopt module and begins the hyperparameter tuning process. Set
    up to work across multiple models and metrics as required.
    Args:
        df_features (pd.DataFrame): Dataframe of features for training.
        df_labels (pd.DataFrame): Dataframe of labels for training.
        metric_name (object): The sklearn (or other) metric to be used to perform evaluation of the model during hyperparameter tuning.
        max_eval_param_tuning (int): The maximum number of iterations to perform with Hyperopt.
        explore_dir_params (int): Explore output directory.
        model_names (list): any in 'XGBoost','lightGBM','ranfomForest'
    Returns:
        best_params (dict): Best params and best model found during hyperparameter tuning.
"""     
    # Set up some variables for hyperopt to consume     

    metric_name, MAX_EVALS, do_cv = metric_name, max_eval_param_tuning, True     
    best_loss, best_params = [], []

    # For each choice of ml model, lets go through the hyperopt process and find the best parameters     
    for model_name in model_names:                 
        objopt = HPOpt(df_features, df_labels, model_name, metric_name, do_cv)         
        trials = SparkTrials() if parallel_tuning else Trials()         
        # Get the best found parameters         
        best = fmin(fn=objopt.cross_validate_with_model, space=objopt.space, algo=tpe.suggest, max_evals=MAX_EVALS,                     trials=trials)         
        best_loss_idx = np.argmin(trials.losses())         
        best_loss.append(trials.results[best_loss_idx]['loss'])         
        best_params.append(trials.results[best_loss_idx]['params'])         
        logger.info(f"Best params: {best_params}")         
        # best_params.append(best_params)     
        # # The best ml model will have the best loss result     
        best_method = np.argmin(best_loss)     
        # Select that method going forwards, and use the best parameters     
        ml_method = model_names[best_method]     
        params = best_params[best_method]     
        logger.info(f"Best params {params}")     
        # Setting up a dataframe of best parameters, to be appended and output every week. Probably a way neater     
        # way to do this, could be re-written     
        df_parameters = pd.DataFrame.from_dict(best_params[best_method], orient='index', columns=['optimalvalue'])     
        df_parameters = df_parameters.reset_index(level=0)     
        df_parameters = df_parameters.rename(columns={'index': 'parameters'})     
        df_parameters = df_parameters.append({'parameters': 'optimalmethod', 'optimalvalue': ml_method}, ignore_index=True)     
        df_parameters = df_parameters.applymap(str)     

    return params

# COMMAND ----------

from typing import Any

import shap
from matplotlib.figure import Figure
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator


def shap_force(
    clf: BaseEstimator,
    index: int,
    X_train_df: DataFrame,
    y_train: Series,
    explainer: shap.Explainer,
    shap_vals: np.ndarray,
    labels: list,
) -> Figure:
    """
    Takes in a fitted classifier Pipeline, the name of the classifier step,
    the X training DataFrame, the y train array, a shap explainer, and the
    shap values to print the ground truth and predicted label and display
    the shap force plot for the record specified by index.

    Args:
        clf (BaseEstimator): A fitted sklearn classifier or Pipeline.
        index (int): The index of the observation of interest.
        X_train_df (DataFrame): A Pandas DataFrame from the train-test-split
            used to train the classifier, with column names corresponding to
            the feature names.
        y_train (Series): Subset of y data used for training.
        explainer (shap.Explainer): A fitted shap.Explainer object.
        shap_vals (np.ndarray): The array of shap values.

    Returns:
        Figure: Shap force plot showing the breakdown of how the model made
            its prediction for the specified record in the training set.
    """
    shap.initjs()

    # Store model prediction and ground truth label
    pred = np.round(clf.predict(X_train_df.loc[X_train_df.index == index])[0])
    true_label = y_train.loc[y_train.index == index].values[0]

    # Assess accuracy of prediction
    accurate = "Correct!" if true_label == int(pred) else "Incorrect"

    # Print output that checks model's prediction against true label
    print("===" * 12)
    print(f"Ground Truth Label: {labels[true_label]}")
    print()
    print(f"Model Prediction: {labels[int(pred)]} -- {accurate}")
    print("===" * 12)
    print()

    # Plot the prediction's explanation
    fig = shap.force_plot(
        explainer.expected_value,
        shap_vals[index, :],
        X_train_df.iloc[index, :],
        matplotlib=True,
    )

    return fig
