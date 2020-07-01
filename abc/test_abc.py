#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/6/23 12:42

@author: tatatingting
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def get_tidy_data(filename, filename2):
    df = pd.read_csv(filename, encoding='utf-8-sig', parse_dates=[0], usecols=[0, 1])  # date, close
    df.sort_values(by='date', ascending=True, inplace=True)
    df['buyin'] = 0
    df.to_csv(filename2, encoding='utf-8-sig', index=False)
    print('---数据已整理完毕')


def update_buy_list(buy_list, date, new_price, buy_in):
    if buy_in:
        buy_list.update({date: [new_price, None, None, None, None, None]})
    for k, v in buy_list.items():
        if date > k and not v[-1]:
            hold_days = date - k
            new_return = round(new_price / v[0] * 100 - 100, 2)
            max_return = max(v[-2], new_return) if v[-2] else new_return
            sell_out = False
            if new_return <= -5:
                sell_out = '止损ge'
            elif (max_return >= 5) and (new_return < (max_return * 0.8)):
                sell_out = '止盈'
            buy_list.update({k: [v[0], date, new_price, hold_days, new_return, max_return, sell_out]})


def get_report(buy_list, filename):
    df = pd.DataFrame.from_dict(buy_list, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['date1', 'buy', 'date2', 'sell', 'days', 'new_return', 'max_return', 'mode']
    filename3 = filename.replace('tidy', 'tidy_out')
    df.to_csv(filename3, encoding='utf-8-sig', index=False)
    print(df)
    df.loc[:, ['buy', 'sell']].plot()
    plt.show()
    df.loc[:, ['max_return', 'new_return']].plot()
    plt.show()


def get_chunk_data(filename):
    filename3 = filename.replace('tidy', 'tidy_2')
    reader = pd.read_csv(filename3, encoding='utf-8-sig', parse_dates=[0], usecols=[0, 1, 2], chunksize=40)  # date, close, buyin
    buy_list = {}
    print('---开始批量执行，请耐心等待')
    for chunk in reader:
        height = chunk.shape[0]
        for i in range(height):
            date = chunk.iloc[i, 0]
            new_price = chunk.iloc[i, 1]
            buy_in = chunk.iloc[i, 2]
            # print(buy_list, date, new_price, buy_in)
            update_buy_list(buy_list, date, new_price, buy_in)
    print('---yes')
    get_report(buy_list, filename)


if __name__ == '__main__':
    flag = True
    while flag:
        try:
            number = input('请输入代码： ')
            filename = os.path.join('data', number, '{}.csv'.format(int(number)))
            filename2 = filename.replace('.csv', '_tidy.csv')
            get_tidy_data(filename, filename2)
            buyin = input('---完成手动标记buyin操作后请输入ok： ')
            if buyin == 'ok':
                get_chunk_data(filename=filename2)
                flag = False
        except:
            print('')
