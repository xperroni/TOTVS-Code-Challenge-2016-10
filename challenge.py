#!/usr/bin/env python3

from collections import OrderedDict
from sys import maxsize
from time import gmtime, mktime, strftime, strptime

from matplotlib import pyplot, cm
from numpy import mean as amean
from numpy import std as astd
from numpy import polyfit, polyval
from numpy import arange, log, zeros, cos, sin
from numpy.random import permutation
from scipy.optimize import curve_fit
from sklearn.covariance import empirical_covariance


def load(path):
    r'''Load data set from file.
    '''
    return eval(''.join(line.strip() for line in open(path)))


def summary(data):
    r'''Collect summary stats about the data set:

        * Number of data entries;
        * Number of distinct product categories;
        * Names of each product category.
    '''
    companies = set()
    labels = OrderedDict()
    clients = dict()
    for entry in data:
        client = entry['infAdic']['infCpl']
        clients[client] = clients.get(client, 0) + 1

        companies.add(entry['emit']['xFant'] + ' (CNPJ: %s)' % entry['emit']['cnpj'])
        for det in entry['dets']:
            labels.setdefault(det['prod']['xProd'], len(labels))

    print('Dataset size: %d entries' % len(data))
    print('Companies (%d): %s' % (len(companies), str(companies)))
    print('Products (%d): %s' % (len(labels), str(labels.keys())))
    print('Clients (%d): %s' % (len(clients), str(clients)))

    # Sort the labels dictionary by order of appearance in the data set
    labels = OrderedDict(sorted(labels.items(), key=lambda t: t[1]))

    return (len(data), len(labels), labels)


def embed(data):
    r'''Convert the given data set into a vector list.

        Each entry is converted to a vector as long as the number of distinct product
        categories. Vector cells are assigned to specific categories through the list
        of category labels returned by the `summary()` function.
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


def datasets(data, f_train=0.7, f_valid=0.2):
    (labels, timestamps, inputs) = embed(data)

    # Randomize data set
    print(timestamps.shape)
    indexes = permutation(timestamps.shape[0])
    timestamps = timestamps[indexes]
    inputs = inputs[indexes]

    (m, n) = inputs.shape
    i_train = int(f_train * m)
    i_valid = i_train + int(f_valid * m)

    train = (timestamps[:i_train], inputs[:i_train])
    valid = (timestamps[i_train:i_valid], inputs[i_train:i_valid])
    tests = (timestamps[i_valid:], inputs[i_valid:])

    return (labels, train, valid, tests)


def plot_covariance(labels, X):
    r'''Draws a color map plot of the covariance matrix of the data set X.

        The main diagonal is zeroed so that covariances between different features are
        highlighted.
    '''
    r = range(len(labels))
    C = empirical_covariance(standardize(X))
    for i in r:
        C[i, i] = 0

    colors = pyplot.matshow(C)
    pyplot.colorbar(colors)

    pyplot.xticks(r, labels.keys(), rotation='vertical')
    pyplot.yticks(r, labels.keys())

    pyplot.show()


def weekly_totals(timestamps, inputs):
    week0 = maxsize
    weeks = 0
    totals = zeros((54,))
    for (timestamp, total) in zip(timestamps, inputs[:, -1].flat):
        week = int(strftime('%U', gmtime(timestamp)))
        week0 = min(week0, week)
        weeks = max(weeks, week + 1)
        totals[week] += total

    return (week0, totals[week0:weeks])


def extrapolate(values, step):
    n = len(values)
    l = n - 1
    w = int(1.0 / step)
    extra = zeros(1 + l * w)
    extra[-1] = values[-1]
    for (i, y1, y2) in zip(range(n), values[:-1], values[1:]):
        b = y2 - y1
        a = y1 - b * i
        for (j, x) in zip(range(i * w, (i + 1) * w), arange(i, i + 1, step)):
            extra[j] = a + b * x

    return extra

def plot_regress(timestamps, inputs, step=1):
    (week0, totals) = weekly_totals(timestamps, inputs)
    weekn = week0 + len(totals)

    x_train = arange(week0, weekn - 1 + step, step)
    x_tests = arange(week0, weekn + step, step)
    y_train = extrapolate(totals, step)

    #p = polyfit(x_train, y_train, 2)

    #f = lambda x, p1, p2: p1 * log(x) + p2
    #(p, cov) = curve_fit(f, x_train, y_train, p0=(1.0, 0.0))

    f = lambda x, a0, a1: a0 + a1 * cos(x)
    (p, cov) = curve_fit(f, x_train, y_train, p0=(0.0, 1.0))

    pyplot.plot(x_train, y_train)
    pyplot.plot(x_tests, f(x_tests, *p))
    #pyplot.plot(x_tests, polyval(p, x_tests))
    pyplot.show()


def main():
    data = load('sample.txt')
    (labels, timestamps, inputs) = embed(data)

    #plot_regress(timestamps, inputs)

    plot_covariance(labels, inputs)
    #totals = weekly_totals(timestamps, inputs)
    #pyplot.plot(totals)
    #pyplot.show()


if __name__ == '__main__':
    main()
