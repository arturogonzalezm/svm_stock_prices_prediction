import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def get_data(file_name):
    """
    :param file_name: Input CSV file
    :return: Date and Price
    """

    with open(file_name, 'r') as csvfile:
        csv_file_reader = csv.reader(csvfile)
        next(csv_file_reader)
        date = []
        price = []
        for row in csv_file_reader:
            date.append(int(row[0].split('-')[0]))
            price.append(float(row[1]))
        return date, price


def predict_price(date, price, x):
    """
    :param date: Dates
    :param price: Prices
    :param x: Day
    :return: RBF model, Linear model and Polynomial model
    """

    '''
    converting to matrix of n X 1
    '''
    date = np.reshape(date, (len(date), 1))
    '''
    defining the support vector regression models
    '''
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    '''
    fitting the data points in the models
    '''
    svr_rbf.fit(date, price)
    svr_lin.fit(date, price)
    svr_poly.fit(date, price)

    '''
    plotting the initial data points
    '''
    plt.scatter(date, price, color='black', label='Data')
    '''
    plotting the line made by the RBF kernel
    '''
    plt.plot(date, svr_rbf.predict(date), color='red', label='RBF model')
    '''
    plotting the line made by linear kernel
    '''
    plt.plot(date, svr_lin.predict(date), color='green', label='Linear model')
    '''
    plotting the line made by polynomial kernel
    '''
    plt.plot(date, svr_poly.predict(date), color='blue', label='Polynomial model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


if __name__ == '__main__':
    dates, prices = get_data('~/svm_stock_prices_prediction/tests/data/AAPL.csv')
    predict_price(dates, prices, 29)
