def prediction(ticker, n_days):
    import pandas as pd
    import yfinance as yf
    import numpy as np
    import plotly.graph_objs as go
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from datetime import date, timedelta
    import warnings
    warnings.filterwarnings("ignore")

    #Download the ticker data and make neww column called Day
    df= yf.download(ticker, period= '60d')
    df.reset_index(inplace=True)
    df["Day"]= df.index
    
    #Define your X/Features and your Y/Labels before you can split.

    Days= list()
    for i in range(len(df.Day)):
        Days.append([i])

    X=Days
    y=df[["Close"]]
    print(X)

    #SPLIT TRAIN AND TEST DATA BEFORE PREPROCESSSING AND STANDARDIZE Xtrain and Xtest Separately.PIPELINE DOES IT AUTOMATICALLY FOR YOU.
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.1, random_state=0)


    #Perform GridSerch to find best parameters for our SVR model
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.001, 0.01, 0.1, 1, 100, 1000],
            'epsilon': [
                0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,
                50, 100, 150, 1000
            ],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]
        },
        cv=5,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1)

    #Flatten our y by using ravel. Ravel converts 2d array to 1d array that the fit method accepts. Fit train values to the gsc.
    y_train = y_train.values.ravel() # or y_train.reshape(-1) returns 1D array also
    y_train
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_

    #Our model now has the best parameters after looking at our data,so let's use it.
    best_svr = SVR(kernel='rbf',
                   C=best_params["C"],
                   epsilon=best_params["epsilon"],
                   gamma=best_params["gamma"],
                   max_iter=-1)

    #Make a Pipeline. It is the surest way to avoid data leakage during preprocessing
    pipeline = make_pipeline(StandardScaler(), best_svr)
    pipeline.fit(X_train, y_train) 
    y_pred = pipeline.predict(X_test)               

    #Print the Error Metrics MAE AND MSE. The closer to 0 the better. But be careful not to Overfit
    print(f'\n The mean absolute error is:', mean_absolute_error(y_test, y_pred)) 
    print(f'\n The mean squared error is:', mean_squared_error(y_test, y_pred))

    #Prepare values for our plot
    output_days = list()
    for i in range(1, n_days):
        output_days.append([i + X_test[-1][0]])

    dates = []
    current = date.today()
    for i in range(n_days):
        current += timedelta(days=1)
        dates.append(current)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,  # np.array(ten_days).flatten(), 
            y=best_svr.predict(output_days),
            mode='lines+markers',
            name='data'))
    fig.update_layout(
        title="Predicted Close Price of next " + str(n_days - 1) + " days",
        xaxis_title="Date",
        yaxis_title="Closed Price",
        legend_title="Legend Title",
    )

    return fig