#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/1/3 23:23

@author: tatatingting
"""


import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def record_init(new_column='up', new_price=0):
    record = {}

    record.update({'down': {'new_price': -np.inf}})
    record.update({'ziup': {'new_price': -np.inf}})
    record.update({'ciup': {'new_price': -np.inf}})
    record.update({'up': {'new_price': np.inf}})
    record.update({'zidown': {'new_price': np.inf}})
    record.update({'cidown': {'new_price': np.inf}})

    record.update({'last': {'column': new_column, 'new_price': new_price}})
    record.get(new_column).update({'new_price': new_price})

    record.update({'key': {
        'point_up': np.inf,
        'sell_point': np.inf,
        'judge_up': np.inf,
        'min_down': np.inf,
        'point_down': -np.inf,
        'buy_point': -np.inf,
        'judge_down': -np.inf,
        'max_up': -np.inf,
        'point_up_from': 'ziup',
        'point_down_from': 'zidown',
    }})

    record.update({'sig': {
        'signal': [],
        'result': [],
    }})

    record.update({'trade': {
        'buy_in_day': False,
        'buy_out_day': False,
        'buy_in': False,
        'buy_price_in': np.nan,
        'buy_price_out': np.nan,
    }})

    return record


def match_rule(new_price, record, s1, s2, s3):
    rule = []
    new_column = ''

    last_column = record.get('last').get('column')
    last_price = record.get('last').get('new_price')
    point_up = record.get('key').get('point_up')
    point_up_from = record.get('key').get('point_up_from')
    point_down = record.get('key').get('point_down')
    point_down_from = record.get('key').get('point_down_from')

    # 开始匹配

    if last_column in ['down', 'zidown', 'cidown']:
        if new_price > last_price + s1:
            rule.append('up_trend')
            if point_up_from == 'up':
                if new_price > point_up:
                    new_column = 'up'
                else:
                    new_column = 'ziup'
            elif point_up_from == 'ziup':
                if last_column == 'down':
                    if new_price > point_up + s2:
                        new_column = 'up'
                    elif new_price > point_up:
                        new_column = 'ziup'
                    elif new_price < point_up - s2:
                        new_column = 'ziup'
                    else:
                        new_column = 'ciup'
                else:
                    if new_price > point_up + s2:
                        new_column = 'up'
                    elif new_price > point_up:
                        new_column = 'ziup'
                    else:
                        new_column = 'ciup'
        else:
            if last_column == 'down':
                if new_price < last_price:
                    new_column = 'down'
            elif last_column == 'zidown':
                if point_down_from == 'down':
                    if new_price < point_down:
                        new_column = 'down'
                    elif new_price < last_price:
                        new_column = 'zidown'
                elif point_down_from == 'zidown':
                    if new_price < point_down - s2:
                        new_column = 'down'
                    elif new_price < last_price:
                        new_column = 'zidown'
            elif last_column == 'cidown':
                if point_down_from == 'zidown':
                    if new_price < point_down - s2:
                        new_column = 'down'
                    elif new_price < point_down:
                        new_column = 'zidown'
                    elif new_price < last_price:
                        new_column = 'cidown'
    elif last_column in ['up', 'ziup', 'ciup']:
        if new_price < last_price - s1:
            rule.append('down_trend')
            if point_down_from == 'down':
                if new_price < point_down:
                    new_column = 'down'
                else:
                    new_column = 'zidown'
            elif point_down_from == 'zidown':
                if last_column == 'up':
                    if new_price < point_down - s2:
                        new_column = 'down'
                    elif new_price < point_down:
                        new_column = 'zidown'
                    elif new_price > point_down + s2:
                        new_column = 'zidown'
                    else:
                        new_column = 'cidown'
                else:
                    if new_price < point_down - s2:
                        new_column = 'down'
                    elif new_price < point_down:
                        new_column = 'zidown'
                    else:
                        new_column = 'cidown'
        else:
            if last_column == 'up':
                if new_price > last_price:
                    new_column = 'up'
            elif last_column == 'ziup':
                if point_up_from == 'up':
                    if new_price > point_up:
                        new_column = 'up'
                    elif new_price > last_price:
                        new_column = 'ziup'
                elif point_up_from == 'ziup':
                    if new_price > point_up + s2:
                        new_column = 'up'
                    elif new_price > last_price:
                        new_column = 'ziup'
            elif last_column == 'ciup':
                if point_up_from == 'ziup':
                    if new_price > point_up + s2:
                        new_column = 'up'
                    elif new_price > point_up:
                        new_column = 'ziup'
                    elif new_price > last_price:
                        new_column = 'ciup'

    # 最后再更新 point & p_from
    if 'up_trend' in rule or 'down_trend' in rule:
        if last_column == 'up':
            record.get('key').update({'point_up': last_price, 'point_up_from': 'up', 'sell_point': last_price})
        elif last_column == 'ziup':
            record.get('key').update({'point_up': last_price, 'point_up_from': 'ziup', 'judge_up': last_price})
        elif last_column == 'down':
            record.get('key').update({'point_down': last_price, 'point_down_from': 'down', 'buy_point': last_price})
        elif last_column == 'zidown':
            record.get('key').update({'point_down': last_price, 'point_down_from': 'zidown', 'judge_down': last_price})

    # 处理匹配结果

    if new_column != '':
        record.get('last').update({'column': new_column, 'new_price': new_price})
        record.get(new_column).update({'new_price': new_price})

    return None


def scale_tidy(sample, mode, p1, p2, p3, p4, p5):
    df_price = pd.read_csv(sample, encoding='utf-8-sig', parse_dates=[0], skiprows=p5)
    df_price.drop_duplicates(keep='last', inplace=True)
    df_price.columns = ['date', 'new_price']
    df_price.set_index(keys='date', inplace=True)
    df_price.sort_index(inplace=True)
    df_price['new_price'] = df_price['new_price'].round(3)

    df_price['1_diff_mean'] = df_price['new_price'].diff().rolling(window=p4).mean().abs().round(3)
    df_price['1_diff_std'] = df_price['new_price'].diff().rolling(window=p4).std().round(3)

    df_price['2_diff_mean'] = df_price['new_price'].diff().abs().rolling(window=p4).mean().round(3)
    df_price['2_diff_std'] = df_price['new_price'].diff().abs().rolling(window=p4).std().round(3)

    df_price['new_price_max'] = df_price['new_price'].rolling(window=p4).max().round(3)
    df_price['new_price_min'] = df_price['new_price'].rolling(window=p4).min().round(3)
    df_price['new_price_max_min'] = df_price['new_price_max'] - df_price['new_price_min']

    df_price['new_price_mean'] = df_price['new_price'].rolling(window=p4).mean().round(3)

    df_price.fillna(method='bfill', inplace=True)

    if mode == 1:
        df_price['s1'] = df_price['1_diff_mean'] + df_price['1_diff_std'] * p1
        df_price['s2'] = df_price['1_diff_mean'] + df_price['1_diff_std'] * p2
        df_price['s3'] = df_price['1_diff_mean'] + df_price['1_diff_std'] * p3
    elif mode == 2:
        df_price['s1'] = df_price['2_diff_mean'] + df_price['2_diff_std'] * p1
        df_price['s2'] = df_price['2_diff_mean'] + df_price['2_diff_std'] * p2
        df_price['s3'] = df_price['2_diff_mean'] + df_price['2_diff_std'] * p3
    elif mode == 3:
        df_price['s1'] = df_price['2_diff_mean'] + df_price['2_diff_std'] * p1
        df_price['s2'] = df_price['2_diff_mean'] + df_price['2_diff_std'] * p2
        df_price['s3'] = df_price['1_diff_mean'] + df_price['1_diff_std'] * p3
    elif mode == 4:
        df_price['s1'] = df_price['2_diff_mean'] * p1
        df_price['s2'] = df_price['2_diff_mean'] * p2
        df_price['s3'] = df_price['2_diff_mean'] * p3
    elif mode == 5:
        df_price['s1'] = df_price['1_diff_mean'] * p1
        df_price['s2'] = df_price['1_diff_mean'] * p2
        df_price['s3'] = df_price['1_diff_mean'] * p3
    elif mode == 6:
        df_price['s1'] = df_price['new_price_max_min'] * p1
        df_price['s2'] = df_price['s1'] * p2
        df_price['s3'] = df_price['new_price'] * p3
    elif mode == 7:
        df_price['s1'] = df_price['new_price_mean'] * p1
        df_price['s2'] = df_price['new_price_mean'] * p2
        df_price['s3'] = df_price['new_price_mean'] * p3

    path_tidy = sample.replace('.', '_tidy.')
    df_price.to_csv(path_tidy, encoding='utf-8-sig')

    return None


def record_update(sample, record, size, mode, p1, p2, p3, p4):
    path_out = sample.replace('.', '_log.')
    col_name = [
        'date',
        'new_price',
        'column',
        'col_price',
        'signal',
        'buy_price_in',
        'buy_price_out',
    ]
    pd.DataFrame(columns=[col_name]).to_csv(path_out, encoding='utf-8-sig', index=False)

    path_tidy = sample.replace('.', '_tidy.')
    reader = pd.read_csv(path_tidy, usecols=['date', 'new_price', 's1', 's2', 's3'], chunksize=size)
    for chunk in reader:
        result = []
        for i in np.arange(chunk.shape[0]):
            new_price = chunk.iloc[i, 1].round(3)
            s1 = chunk.iloc[i, 2].round(3)
            s2 = chunk.iloc[i, 3].round(3)
            s3 = chunk.iloc[i, 4].round(3)
            match_rule(new_price, record, s1, s2, s3)
            result.append([
                chunk.iloc[i, 0],
                new_price,
                record.get('last').get('column'),
                record.get('last').get('new_price'),
                record.get('sig').get('signal'),
                record.get('key').get('buy_point'),
                record.get('key').get('sell_point'),
            ])
        print('/', end='')
        df_result = pd.DataFrame(result)
        df_result.to_csv(path_out, mode='a', encoding='utf-8-sig', index=False, header=False)
        back_result(sample, mode, p1, p2, p3, p4)
    return None


def back_result(sample, mode, p1, p2, p3, p4):
    path_log = sample.replace('.', '_log.')
    result = pd.read_csv(path_log)

    re1 = result.loc[:, ['date', 'column', 'col_price']].pivot('date', 'column', 'col_price').reset_index()
    re1 = re1.loc[:, ['date', 'ciup', 'ziup', 'up', 'down', 'zidown', 'cidown']]
    re2 = result.loc[:, ['date', 'new_price', 'signal', 'buy_price_in', 'buy_price_out']]
    result = pd.merge(re1, re2)
    result.to_excel(path_log.replace('.csv', '.xlsx'), encoding='utf-8-sig')
    result.set_index('date', inplace=True)

    # Index(['date', 'ciup', 'ziup', 'up', 'down', 'zidown', 'cidown', 'new_price',
    #        'signal', 'buy_price_in', 'buy_price_out'],
    #       dtype='object')

    # price
    title = path_log.replace('.csv', '.png')
    fig, axes = plt.subplots(figsize=(20, 10))
    result['new_price'].plot(marker='.', color='grey', legend=True, alpha=0.2, ax=axes)
    # cols price
    result['ciup'].plot(marker='.', color='yellow', legend=True, ax=axes)
    result['ziup'].plot(marker='.', color='orange', legend=True, ax=axes)
    result['up'].plot(marker='.', color='red', legend=True, ax=axes)
    result['down'].plot(marker='.', color='green', legend=True, ax=axes)
    result['zidown'].plot(marker='.', color='cornflowerblue', legend=True, ax=axes)
    result['cidown'].plot(marker='.', color='springgreen', legend=True, ax=axes)

    # trade sig
    # result['buy_price_in'].plot(marker='+', color='r', legend=True, alpha=0.5, ax=axes)
    # result['buy_price_out'].plot(marker='_', color='g', legend=True, alpha=0.5, ax=axes, grid=True)

    # # tip
    # t_list = [
    #     ['b', 'buy_price_in', 'r', 1],
    #     ['s', 'buy_price_out', 'g', 1.5],
    # ]
    # for i in t_list:
    #     for j in np.arange(result[i[1]].shape[0]):
    #         if result[i[1]][j] > 0:
    #             axes.annotate(
    #                 '{}:{}'.format(i[0], result[i[1]][j].round(2)),
    #                 xy=(j, result[i[1]][j]),
    #                 xytext=(j, result[i[1]][j] - i[3]),
    #                 fontsize=16,
    #                 color='{}'.format(i[2]),
    #             )

    title = title.replace('_log', '_log_{}_{}_{}_{}_{}'.format(mode, p1, p2, p3, p4))
    axes.set_title(title)
    plt.savefig(title)
    plt.show()
    plt.close(fig)

    return None


def run_it(sample, mode, p1, p2, p3, p4, p5, r, size):
    scale_tidy(sample, mode, p1, p2, p3, p4, p5)
    record_update(sample, r, size, mode, p1, p2, p3, p4)
    # back_result(sample)

    return None


if __name__ == '__main__':
    s_time = time.time()

    # if mode == 1:
    #     df_price['s1'] = df_price['1_diff_mean'] + df_price['1_diff_std'] * p1
    #     df_price['s2'] = df_price['1_diff_mean'] + df_price['1_diff_std'] * p2
    #     df_price['s3'] = df_price['1_diff_mean'] + df_price['1_diff_std'] * p3
    # elif mode == 2:
    #     df_price['s1'] = df_price['2_diff_mean'] + df_price['2_diff_std'] * p1
    #     df_price['s2'] = df_price['2_diff_mean'] + df_price['2_diff_std'] * p2
    #     df_price['s3'] = df_price['2_diff_mean'] + df_price['2_diff_std'] * p3
    # elif mode == 3:
    #     df_price['s1'] = df_price['2_diff_mean'] + df_price['2_diff_std'] * p1
    #     df_price['s2'] = df_price['2_diff_mean'] + df_price['2_diff_std'] * p2
    #     df_price['s3'] = df_price['1_diff_mean'] + df_price['1_diff_std'] * p3
    # elif mode == 4:
    #     df_price['s1'] = df_price['2_diff_mean'] * p1
    #     df_price['s2'] = df_price['2_diff_mean'] * p2
    #     df_price['s3'] = df_price['2_diff_mean'] * p3
    # elif mode == 5:
    #     df_price['s1'] = df_price['1_diff_mean'] * p1
    #     df_price['s2'] = df_price['1_diff_mean'] * p2
    #     df_price['s3'] = df_price['1_diff_mean'] * p3
    # elif mode == 6:
    #     df_price['s1'] = df_price['new_price_max_min'] * p1
    #     df_price['s2'] = df_price['s1'] * p2
    #     df_price['s3'] = df_price['new_price'] * p3
    # elif mode == 7:
    #     df_price['s1'] = df_price['new_price_mean'] * p1
    #     df_price['s2'] = df_price['new_price_mean'] * p2
    #     df_price['s3'] = df_price['new_price_mean'] * p3

    paras = [
        [1, 0.9, 0.5, 0.01, 21],
        [1, 0.9, 0.5, 0.05, 21],
        [1, 0.9, 0.5, 0.01, 5],
        [1, 0.9, 0.5, 0.05, 5],
        [2, 0.9, 0.5, 0.01, 21],
        [2, 0.9, 0.5, 0.01, 5],
        [3, 0.9, 0.5, 0.05, 5],
        [4, 6, 3, 1, 5],

    ]

    for para in paras:
        run_it('601519_1819.csv', mode=para[0], p1=para[1], p2=para[2], p3=para[3], p4=para[4], p5=0, r=record_init(), size=600)
    print(time.time() - s_time)
