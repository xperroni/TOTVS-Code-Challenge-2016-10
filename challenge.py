#!/usr/bin/env python3

from collections import OrderedDict
from sys import maxsize
from time import gmtime, mktime, strftime, strptime

from matplotlib import pyplot, cm
from numpy import mean as amean
from numpy import std as astd
from numpy import arange, zeros, sin
from scipy.optimize import curve_fit
from sklearn.covariance import empirical_covariance


def load(path):
    r'''Load dataset from file.
    '''
    return eval(''.join(line.strip() for line in open(path)))


def dataset_characteristics(data):
    r'''Print a quick report of dataset features.
    '''
    companies = set()
    products = set()
    clients = set()
    for entry in data:
        client = entry['infAdic']['infCpl']
        clients.add(client)

        companies.add(entry['emit']['xFant'] + ' (CNPJ: %s)' % entry['emit']['cnpj'])
        for det in entry['dets']:
            products.add(det['prod']['xProd'])

    print('Dataset size: %d entries' % len(data))
    print('Companies (%d): %s' % (len(companies), str(companies)))
    print('Products (%d): %s' % (len(products), str(products)))
    print('Clients (%d): %s' % (len(clients), str(clients)))


def summary(data):
    r'''Collect summary stats about the dataset:

        * Number of data entries;
        * Number of distinct product categories;
        * Names of each product category.
    '''
    labels = OrderedDict()
    for entry in data:
        for det in entry['dets']:
            labels.setdefault(det['prod']['xProd'], len(labels))

    # Sort the labels dictionary by order of appearance in the dataset
    labels = OrderedDict(sorted(labels.items(), key=lambda item: item[1]))

    return (len(data), len(labels), labels)


def embed(data):
    r'''Convert dataset entries into vectors.

        Entries are converted to vectors of item prices, plus the total price at the
        last cell. Therefore, vectors are as long as the number of distinct item
        categories plus one. The item represented by a given vector cell is determined
        through the list of item labels returned by the `summary()` function.

        Additionally, this function returns a vector of timestamps corresponding to the
        creation time of each entry. The item label list is also returned for use by
        client code, updated with the entry for the total price.
    '''
    (m, n, labels) = summary(data)
    inputs = zeros((m, n + 1))
    timestamps = zeros(m)
    for (i, entry) in enumerate(data):
        timestamps[i] = mktime(strptime(entry['ide']['dhEmi']['$date'], r'%Y-%m-%dT%H:%M:%S.000Z'))
        inputs[i, -1] = entry['complemento']['valorTotal']
        for row in entry['dets']:
            label = row['prod']['xProd']
            j = labels[label]
            inputs[i, j] = row['prod']['vProd']

    labels['TOTAL'] = n

    return (labels, timestamps, inputs)


def standardize(inputs):
    r'''Standardize features into unitless quantities for better inter-feature comparison.
    '''
    u_inputs = amean(inputs, axis=0)
    s_inputs = astd(inputs, axis=0)
    return (inputs - u_inputs) / s_inputs


def plot_covariance(data):
    r'''Draws a color map plot of the covariance matrix of the dataset X.

        The main diagonal is zeroed so that covariances between different features are
        highlighted.
    '''
    (labels, timestamps, inputs) = embed(data)
    r = range(len(labels))
    cov = empirical_covariance(standardize(inputs))
    for i in r:
        cov[i, i] = 0

    colors = pyplot.matshow(cov)
    pyplot.colorbar(colors)

    pyplot.xticks(r, labels.keys(), rotation='vertical')
    pyplot.yticks(r, labels.keys())

    pyplot.show()


def daily_totals(timestamps, inputs):
    r'''Compute weekly price totals for the list of input vectors.

        Returns the index of the first week in the dataset (0-based) and the list of
        weekly totals.
    '''
    day0 = maxsize
    days = 0
    totals = zeros((366,))
    for (timestamp, total) in zip(timestamps, inputs[:, -1].flat):
        day = int(strftime('%j', gmtime(timestamp)))
        day0 = min(day0, day)
        days = max(days, day + 1)
        totals[day] += total

    return (day0, totals[day0:days])


def weekly_totals(timestamps, inputs):
    r'''Compute weekly price totals for the list of input vectors.

        Returns the index of the first week in the dataset (0-based) and the list of
        weekly totals.
    '''
    week0 = maxsize
    weeks = 0
    totals = zeros((54,))
    for (timestamp, total) in zip(timestamps, inputs[:, -1].flat):
        week = int(strftime('%U', gmtime(timestamp)))
        week0 = min(week0, week)
        weeks = max(weeks, week + 1)
        totals[week] += total

    return (week0, totals[week0:weeks])


def regress(data):
    (labels, timestamps, inputs) = embed(data)
    (x0, y_train) = daily_totals(timestamps, inputs)
    xn = x0 + len(y_train)

    x_train = arange(x0, xn)
    x_tests = arange(x0, xn + 7)

    offset = y_train.mean()
    amplitude = 0.5 * y_train.max()
    f = lambda x, phase: offset + amplitude * sin(x + phase)
    ((phase,), cov) = curve_fit(f, x_train, y_train, p0=(0.0,))

    return (x_train, y_train, x_tests, f(x_tests, phase))


def accumulate_weekly(X, Y):
    y_weekly = 0
    X_weekly = []
    Y_weekly = []
    for (x, y) in zip(X, Y):
        y_weekly += y
        if x % 7 == 0:
            X_weekly.append(x // 7)
            Y_weekly.append(y_weekly)
            y_weekly = 0

    return (X_weekly, Y_weekly)


def plot_regress_daily(data):
    (x_train, y_train, x_tests, y_tests) = regress(data)
    pyplot.plot(x_train, y_train, label='Sample data')
    pyplot.plot(x_tests, y_tests, label='Regression')
    pyplot.legend(bbox_to_anchor=(1, 1.2))
    pyplot.xlabel('Day', labelpad=10)
    pyplot.ylabel('Total sales', labelpad=20)

    print("Days covered: %d" % len(x_train))


def plot_regress_weekly(data):
    (x_train, y_train, x_tests, y_tests) = regress(data)
    (x_train, y_train) = accumulate_weekly(x_train, y_train)
    (x_tests, y_tests) = accumulate_weekly(x_tests, y_tests)
    pyplot.plot(x_train, y_train, label='Sample data')
    pyplot.plot(x_tests, y_tests, label='Regression')
    pyplot.legend(bbox_to_anchor=(1, 1.2))
    pyplot.xlabel('Week', labelpad=10)
    pyplot.ylabel('Total sales', labelpad=20)

    print('Sales forecast for next week: %.2f' % y_tests[-1])


def main():
    data = load('sample.txt')

    dataset_characteristics(data)

    plot_regress_daily(data)

    pyplot.figure()

    plot_regress_weekly(data)

    plot_covariance(data)

    pyplot.show()


if __name__ == '__main__':
    main()
