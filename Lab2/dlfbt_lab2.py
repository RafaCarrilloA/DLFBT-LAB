# ===============================================================================
# DLFBT 2026/2027
# Lab assignment 2
# Authors:
#   RAFAEL CARRILLO ARROYO 31878112C
#   AMIRHOSSEIN ROODAKI Z64987649
# ===============================================================================

import numpy as np
import tensorflow as tf
from time import perf_counter, time
from dataclasses import dataclass
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def set_reproducible(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_phoneme():
    """
    Load the phoneme dataset from the follwing directory:
    "../data/lab2/phoneme.csv"

    The CSV contains:
    - several feature columns;
    - one final binary target column.

    Returns
    -------
    X:
        Feature matrix of shape (n_samples, n_features).
    y:
        Binary target vector of shape (n_samples,).

    TODO
    ----
    Load the dataset and split the last column from the remaining columns.
    """

    # TODO: load the CSV.
    dataset = np.loadtxt("../data/lab2/phoneme.csv", delimiter=",")

    # TODO:
    X = dataset[:, :-1]
    y = dataset[:, -1]

    # TODO: perform a shape / validity check.
    # assert X.ndim == 2
    if X.ndim != 2:
        raise ValueError()
    # assert y.ndim == 1
    if y.ndim != 1:
        raise ValueError()
    # assert len(X) == len(y)
    if len(X) != len(y):
        raise ValueError()
    # assert X.shape[1] == 5
    if X.shape[1] != 5:
        raise ValueError()
    # assert set(np.unique(y)) == {0, 1}
    if set(np.unique(y)) != {0, 1}:
        raise ValueError()

    return X, y


def dataset_overview(X, y):
    labels, counts = np.unique(y, return_counts=True)
    return {
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "class_distribution": {int(k): int(v) for k, v in zip(labels, counts)},
        "feature_mean": np.mean(X, axis=0),
        "feature_std": np.std(X, axis=0),
    }


@dataclass
class DataSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    scaler: Optional[StandardScaler]


def prepare_data(
    X,
    y,
    *,
    test_size=0.20,
    val_size=0.20,
    normalize=True,
    random_state=RANDOM_STATE,
):
    """
    Create stratified training, validation and test partitions.

    Important note:

        FIT THE SCALER USING ONLY THE TRAINING DATA.

    Note that;
    ----
    If information from validation or test samples is used to compute
    preprocessing statistics, information leaks into the training pipeline.
    That makes the reported evaluation too optimistic.

    Parameters
    ----------
    X, y:
        Full dataset.
    test_size:
        Fraction of the full dataset assigned to the test partition.
    val_size:
        Fraction of the full dataset assigned to validation.
    normalize:
        If True, standardize features using StandardScaler.
    random_state:
        Seed passed to train_test_split.

    Returns
    -------
    DataSplit
        Object containing the three partitions and the fitted scaler.

    TODO
    ----
    Implement the full preprocessing pipeline.
    """

    # TODO: validate test_size and val_size.
    if not (0 < test_size < 1) or not (0 < val_size < 1):
        raise ValueError()
    if test_size + val_size >= 1:
        raise ValueError()
    if test_size + val_size > 0.5:
        raise ValueError()

    # TODO: first split -> train+validation and test.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # TODO: compute the validation fraction relative to train+validation.
    val_fraction = val_size / (1 - test_size)

    # TODO: second split -> train and validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction,
        stratify=y_train_val,
        random_state=random_state,
    )

    # TODO: optionally fit StandardScaler ONLY on X_train.
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # TODO: return DataSplit(...)
    return DataSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
    )


def build_baseline_model(input_dim, hidden_units=8):
    """
    Build the small baseline multilayer perceptron used in Part 1.

    Architecture
    ------------
    Input
      -> Dense(hidden_units, activation="relu")
      -> Dense(1, activation="sigmoid")

    Important
    ---------
    This function should BUILD the model, but it should NOT compile it.

    TODO
    ----
    Create and return the Keras Sequential model.
    """

    # TODO:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


def compile_binary_model(
    model,
    optimizer="adam",
    learning_rate=None,
):
    """
    Compile a Keras model for binary classification.

    Required settings
    -----------------
    loss:
        "binary_crossentropy"

    metric:
        "accuracy"

    optimizer:
        May be either:
        - a ready-created Keras optimizer object; or
        - one of the supported strings:
              "sgd"
              "adagrad"
              "rmsprop"
              "adam"

    TODO
    ----
    Compile the model and return it.
    """

    # TODO:
    # If optimizer is a string AND learning_rate is provided,
    # create the appropriate tf.keras.optimizers.* object.
    if isinstance(optimizer, str) and learning_rate is not None:
        if optimizer.lower() == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer.lower() == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer.lower() == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            raise ValueError()

    # TODO:
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def make_early_stopping(patience=20):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )


def train_model(
    model,
    split,
    *,
    epochs=250,
    batch_size=32,
    patience=20,
    verbose=0,
):
    """
    Train a model using the explicit training and validation partitions.

    ------------------
    Use:

        split.X_train
        split.y_train

    for fitting and:

        split.X_val
        split.y_val

    as validation_data.

    Also include the EarlyStopping callback returned by
    `make_early_stopping(patience)`.

    Returns
    -------
    history:
        The Keras History object returned by `model.fit`.

    TODO
    ----
    Call `model.fit(...)` with the appropriate arguments.
    """

    # TODO:
    history = model.fit(
        split.X_train,
        split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[make_early_stopping(patience)],
        verbose=verbose,
    )

    return history


def evaluate_model(model, split):
    """
    Evaluate the trained model ONCE on the held-out test set.

    The returned dictionary must contain:

        "test_loss"
        "test_accuracy"

    Use:
        split.X_test
        split.y_test

    TODO
    ----
    Evaluate the model and return the metrics as normal Python floats.
    """

    # TODO:
    loss, accuracy = model.evaluate(split.X_test, split.y_test)
    return {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
    }


def build_improved_model(input_dim):
    """
    Build a modestly improved network.

    Required architecture
    ---------------------
    Input
      -> Dense(32, relu)
      -> Dropout(0.15)
      -> Dense(16, relu)
      -> Dense(1, sigmoid)

    Compile it with Adam and a learning rate of 1e-3.

    The intention is NOT to build a huge model. We want a small, controlled
    change that can be compared with the baseline.

    TODO
    ----
    Build, compile and return the model.
    """

    # TODO: build the architecture.
    improved_model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # TODO: compile it, preferably by reusing compile_binary_model(...).
    compile_binary_model(improved_model, optimizer="adam", learning_rate=1e-3)

    return improved_model


def build_regularized_model(input_dim, l2_strength=1e-4):
    """
    Build an MLP using L2 kernel regularization.

    Required architecture
    ---------------------
    Dense(32, relu, kernel_regularizer=L2)
      -> Dense(16, relu, kernel_regularizer=L2)
      -> Dense(1, sigmoid)

    Important
    ---------
    This function only builds the model. The notebook will compile it later.

    TODO
    ----
    Create the L2 regularizer and apply it to both hidden Dense layers.
    """

    # TODO:
    reg = tf.keras.regularizers.l2(l2_strength)

    # TODO: build and return the Sequential model.
    regularized_model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    return regularized_model


def build_dropout_model(input_dim, rate=0.25):
    """
    Build an MLP that uses dropout.

    Required architecture
    ---------------------
    Dense(32, relu)
      -> Dropout(rate)
      -> Dense(16, relu)
      -> Dense(1, sigmoid)

    Important
    ---------
    This function only builds the model. The notebook will compile it later.

    TODO
    ----
    Create and return the model.
    """

    # TODO: build and return the Sequential model.
    dropout_model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    return dropout_model


def optimizer_from_name(name, learning_rate=1e-3):
    """
    Construct one of the optimizers studied in Part 2.

    Supported names
    ---------------
    "sgd"
        Standard stochastic gradient descent.

    "momentum"
        SGD with momentum=0.9.

    "nesterov"
        SGD with momentum=0.9 and nesterov=True.

    "adagrad"
        AdaGrad.

    "rmsprop"
        RMSprop.

    "adam"
        Adam.

    All optimizers must use the provided learning rate.

    Hints
    -----
    You may normalize the input string using:

        name.lower()
        .replace(" ", "")
        .replace("-", "")

    Raise ValueError for an unknown optimizer.

    TODO
    ----
    Implement the optimizer selection logic.
    """

    # TODO: return the matching tf.keras.optimizers optimizer.
    name = name.lower().replace(" ", "").replace("-", "")
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif name == "momentum":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif name == "nesterov":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9, nesterov=True
        )
    elif name == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError()


def run_optimizer_experiment(
    split,
    optimizer_name,
    *,
    epochs=150,
    batch_size=32,
    learning_rate=1e-3,
    patience=15,
    seed=RANDOM_STATE,
):
    """
    Train ONE optimizer under controlled conditions.

    Fair-comparison requirements
    ----------------------------
    Keep these fixed:
    - data split;
    - architecture;
    - seed;
    - batch size;
    - stopping rule.

    Only the optimizer should change.


    The notebook expects:

        "optimizer"
        "epochs_run"
        "seconds"
        "test_loss"
        "test_accuracy"
        "history"
        "model"

    TODO
    ----
    Implement one complete experiment.
    """

    # TODO: make the run reproducible.
    set_reproducible(seed)

    # TODO: build model.
    model = build_baseline_model(input_dim=split.X_train.shape[1])

    # TODO: construct optimizer.
    optimizer = optimizer_from_name(optimizer_name, learning_rate=learning_rate)

    # TODO: compile model.
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # TODO: record start time.
    start_time = time()

    # TODO: train.
    history = train_model(
        model,
        split,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )

    # TODO: compute elapsed time.
    elapsed_time = time() - start_time

    # TODO: evaluate.
    evaluation = evaluate_model(model, split)

    # TODO: return the result dictionary.
    return {
        "optimizer": optimizer_name,
        "epochs_run": len(history.history["loss"]),
        "seconds": elapsed_time,
        "test_loss": evaluation["test_loss"],
        "test_accuracy": evaluation["test_accuracy"],
        "history": history.history,
        "model": model,
    }


def compare_optimizers(
    split,
    names=(
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "adam",
    ),
    **kwargs,
):
    """
    Run `run_optimizer_experiment` for every requested optimizer.

    Parameters
    ----------
    split:
        The common DataSplit used for every experiment.
    names:
        Iterable containing optimizer names.
    **kwargs:
        Additional keyword arguments forwarded to run_optimizer_experiment.

    Returns
    -------
    list
        One result dictionary per optimizer.

    TODO
    ----
    A compact loop or list comprehension is sufficient.
    """

    # TODO: call run_optimizer_experiment once per optimizer name.
    return [run_optimizer_experiment(split, name, **kwargs) for name in names]


def available_devices():
    return {
        "CPU": [d.name for d in tf.config.list_physical_devices("CPU")],
        "GPU": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
