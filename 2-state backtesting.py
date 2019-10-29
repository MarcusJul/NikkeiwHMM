# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.mixture as mix

data = pd.read_excel('N225.xlsx', sheet_name='N225').dropna()

dates = data['Date'].values
opens = data['Open'].values
closes = data['Close'].values
highs = data['High'].values
lows = data['Low'].values
adjcloses = data['Adj Close'].values

# The factors:
opclsp = ((data['Open']-data['Close'])/data['Open']).values*100# % change in price during the day
highlowsp = ((data['High']-data['Low'])/data['Close']).values*100 # Spread of high low
prc_ret = data['Close'].pct_change().fillna(0)*100 # Daily price change
data['Return Daily'] = prc_ret
Vol_change = data['Volume'].pct_change().fillna(0)*100
Vol_change[Vol_change==np.inf] = 0
log_ret = np.log((prc_ret+100)/100)

def get_rolling_vol(vol_wind,lst):
    vol_lst = []
    for i in range(len(lst)):
        if i<vol_wind:
            vol_lst.append(0)
        else:
            vol_lst.append(np.std(lst[i-vol_wind:i]))
    return np.array(vol_lst)
#vol30 = get_rolling_vol(30, data['Close'])
#vol100 = get_rolling_vol(100, prc_ret)
#vol250 = get_rolling_vol(250, data['Close'])
#vol500 = get_rolling_vol(500, data['Close'])
#vol1000 = get_rolling_vol(1000, data['Close'])
    
slices = [int(len(data)*0.2),int(len(data)*0.4),int(len(data)*0.6),int(len(data)*0.8),len(data)-1]

# Slice off according to the index and return the rest
def slicer(array, test_p):
    a_shape = array.shape
    head = array[0:test_p[0]]
    tail = array[test_p[1]:]
    ret = np.append(head,tail)
    return ret.reshape(int(len(ret)/a_shape[1]),a_shape[1])

to_float = lambda x : [round(x[i],4) for i in range(len(x))]

# structuring regression data
def structuring_data(test_p,
                     window=0,
                     vollst=0):
    hl = highlowsp[window:]
    oc = opclsp[window:]
    pr = prc_ret[window:]
    date = dates[window:]
    vc = Vol_change[window:]
    lr = log_ret[window:]
    #vol = vollst[window:]
    log_diff = np.append([0],np.diff(lr))
    x = np.column_stack([lr])
    lengthx = len(x)
    #x_train, x_test, dates_train, dates_test = model_selection.train_test_split(x, dates[window:], test_size = 0.2, random_state = 2019)
    x_train, x_test, dates_train, dates_test = slicer(x,test_p), x[test_p[0]:test_p[1]], np.append(date[0:test_p[0]], date[test_p[1]:]), date[test_p[0]:test_p[1]]
    #return x_train
    model = mix.GaussianMixture(n_components=2,covariance_type="full")
    model.fit(x_train)
    train_hiddens = model.predict(x_train)
    print('Training Period Variances:')
    print(np.diag(model.covariances_[0]))
    print(np.diag(model.covariances_[1]))
    
    hiddens = model.predict(x_test)
    print('Score for test:',model.score(x_test))
    print('Score for training:',model.score(x_train))
    
    print('States 1: %d. States 2: %d.'%(len(hiddens[hiddens==0]),len(hiddens[hiddens==1])))
    np.set_printoptions(suppress=True)
    print('The means of state 0:')
    print(model.means_[0])
    print('The means of state 1:')
    print(model.means_[1])

    print(len(x_test),len(dates_test))
    closes_train = np.append(closes[0:test_p[0]], closes[test_p[1]:])
    closes_test = closes[test_p[0]:test_p[1]]
    lr_train = np.append(lr[0:test_p[0]], lr[test_p[1]:])
    lr_test = np.array(lr[test_p[0]:test_p[1]])
        
    print(len(train_hiddens),len(closes_train),len(dates_train),len(lr_train),len(hiddens),len(closes_test),len(dates_test),len(lr_test))
    
    return [train_hiddens, dates_train, closes_train, lr_train, 
            hiddens, dates_test, closes_test, lr_test, 
            [np.diag(model.covariances_[0]),np.diag(model.covariances_[1])]]

def plot_scatter(ret, wind, vollst, highvol_mark, 
                 testing=True):
    fig = plt.figure()
    fig.set_size_inches((18.5,10.5))
    if testing:
        for i in range(wind,len(ret[4])):
            if ret[4][i]==highvol_mark:
                plt.scatter(ret[7][i], vollst[i], color='r', s=4)
            else:
                plt.scatter(ret[7][i], vollst[i], color='g', s=4)
        plt.title('Testing Period LogReturn with Vol')
                
    else: #Training
        for i in range(wind,len(ret[0])):
            if ret[0][i]==highvol_mark:
                plt.scatter(ret[3][i], vollst[i], color='r', s=2)
            else:
                plt.scatter(ret[3][i], vollst[i], color='g', s=2)
        plt.title('Training Period LogReturn with Vol')
    plt.show()
    return

# structuring regression data
def backtesting(test_p,
                     window=0,
                     vollst=0):
    #hl = highlowsp[window:]
    #oc = opclsp[window:]
    #pr = prc_ret[window:]
    date = dates[window:]
    #vc = Vol_change[window:]
    lr = log_ret[window:]
    #vol = vollst[window:]
    #log_diff = np.append([0],np.diff(lr))
    x = np.column_stack([lr])
    #x_train, x_test, dates_train, dates_test = model_selection.train_test_split(x, dates[window:], test_size = 0.2, random_state = 2019)
    x_train, x_test, dates_train, dates_test = slicer(x,test_p), x[test_p[0]:test_p[1]], np.append(date[0:test_p[0]], date[test_p[1]:]), date[test_p[0]:test_p[1]]
    #return x_train
    model = mix.GaussianMixture(n_components=2,covariance_type="full")
    ret = []
    model.fit(x_train)

    train_hiddens = model.predict(x_train)
    ret.append(train_hiddens)
    pps = []
    pps.append(model.predict_proba(x_test))
    ppsr = [] #rolling 
    for i in range(len(x_test)):
        test_p2 = [test_p[0]+i,test_p[1]+i]
        x_train, x_test = slicer(x,test_p2), x[test_p2[0]:test_p2[1]] 
        dates_train, dates_test = np.append(date[0:test_p2[0]], date[test_p2[1]:]), date[test_p2[0]:test_p2[1]] 
        model.fit(x_train)
        #ppsr.append([model.predict(x_test[0].reshape(1,-1)), model.predict_proba(x_test[0].reshape(1,-1))])
        ppsr.append(model.predict_proba(x_test[0].reshape(1,-1)))
    
    pps.append(ppsr)
    return pps
    print('Training Period Variances:')
    print(np.diag(model.covariances_[0]))
    print(np.diag(model.covariances_[1]))
    
    hiddens = model.predict(x_test)
    print('Score for test:',model.score(x_test))
    print('Score for training:',model.score(x_train))
    
    print('States 1: %d. States 2: %d.'%(len(hiddens[hiddens==0]),len(hiddens[hiddens==1])))
    np.set_printoptions(suppress=True)
    print('The means of state 0:')
    print(model.means_[0])
    print('The means of state 1:')
    print(model.means_[1])

    print(len(x_test),len(dates_test))
    closes_train = np.append(closes[0:test_p[0]], closes[test_p[1]:])
    closes_test = closes[test_p[0]:test_p[1]]
    lr_train = np.append(lr[0:test_p[0]], lr[test_p[1]:])
    lr_test = np.array(lr[test_p[0]:test_p[1]])
        
    print(len(train_hiddens),len(closes_train),len(dates_train),len(lr_train),len(hiddens),len(closes_test),len(dates_test),len(lr_test))
    
    return [train_hiddens, dates_train, closes_train, lr_train, 
            hiddens, dates_test, closes_test, lr_test, 
            [np.diag(model.covariances_[0]),np.diag(model.covariances_[1])]]