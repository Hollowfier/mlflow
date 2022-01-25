
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

if __name__ == "__main__":

    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    score = r2_score(diabetes_y_test, diabetes_y_pred)
    error = mean_squared_error(diabetes_y_test, diabetes_y_pred)
    print("Mean squared error: %.2f" % error)
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % score)

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
    # mlflow.log_metric("coefficient", score)
    # mlflow.log_metric("error", error)
    # mlflow.sklearn.log_model(regr, "model")
    # print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
