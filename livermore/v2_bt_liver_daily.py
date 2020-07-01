#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/12/18 16:08

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
        'count_all': 0,
    }})

    record.update({'sig': {
        'rule': [],
        'signal': [],
        'return_r_daily': [],
        'result': [],
    }})

    record.update({'trade': {
        'buy_in_day': False,
        'buy_out_day': False,
        'buy_in': False,
        'buy_price_in': np.nan,
        'buy_price_out': np.nan,
        'position': 0,
        'money': 10000,
        'equity': 1,
        'equity_max': 1,
        'return_r': 0,
        'draw_r': 0,
        'max_draw_r': 0,
        'sharp_r': np.nan,

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

    record.get('key').update({'count_all': record.get('key').get('count_all') + 1})
    record.get('sig').update({'rule': rule})

    if new_column != '':
        record.get('last').update({'column': new_column, 'new_price': new_price})
        record.get(new_column).update({'new_price': new_price})

    # 匹配信号

    signal = []

    last_column = record.get('last').get('column')
    buy_point = record.get('key').get('buy_point')
    sell_point = record.get('key').get('sell_point')
    judge_up = record.get('key').get('judge_up')
    judge_down = record.get('key').get('judge_down')

    max_up = record.get('key').get('max_up')
    min_down = record.get('key').get('min_down')

    if last_column in ['up', 'ziup', 'ciup']:
        record.get('key').update({'max_up': max(new_price, max_up)})
        record.get('key').update({'min_down': np.inf})
    else:
        record.get('key').update({'min_down': min(new_price, min_down)})
        record.get('key').update({'max_up': -np.inf})

    if abs(new_price - sell_point) < s3:
        signal.append('sell_near')
    if abs(new_price - judge_up) < s3:
        signal.append('whether_to_up')
    if new_price > sell_point + s2:
        signal.append('positively_up_resumed')
    if new_price < max_up and max_up > sell_point - s2 > new_price:
        signal.append('indicate_up_over')

    if abs(new_price - buy_point) < s3:
        signal.append('buy_near')
    if abs(new_price - judge_down) < s3:
        signal.append('whether_to_down')
    if new_price < buy_point - s2:
        signal.append('positively_down_resumed')
    if new_price > min_down and min_down < buy_point + s2 < new_price:
        signal.append('indicate_down_over')

    record.get('sig').update({'signal': signal})

    # sim trade

    record.get('trade').update({'buy_price_in': np.nan})
    record.get('trade').update({'buy_price_out': np.nan})

    # buy-in
    if record.get('trade').get('buy_in_day'):
        record.get('trade').update({'buy_in_day': False})
        record.get('trade').update({'buy_in': True})
        record.get('trade').update({'buy_price_in': new_price})
        record.get('trade').update({'position': round(record.get('trade').get('money') / new_price, 3)})
        record.get('trade').update({'money': 0})

    if record.get('trade').get('buy_out_day'):
        record.get('trade').update({'buy_out_day': False})
        record.get('trade').update({'buy_in': False})
        record.get('trade').update({'buy_price_out': new_price})
        record.get('trade').update({'money': round(record.get('trade').get('position') * new_price, 3)})
        record.get('trade').update({'position': 0})

    buy_in = record.get('trade').get('buy_in')
    signal = record.get('sig').get('signal')

    # in
    if not buy_in and ('buy_near' in signal):
        record.get('trade').update({'buy_in_day': True})
    if buy_in and (('indicate_up_over' in signal) or ('sell_near' in signal) or ('positively_down_resumed' in signal)):
        record.get('trade').update({'buy_out_day': True})

    # calculate_trade
    position = record.get('trade').get('position')
    money = record.get('trade').get('money')

    # equity
    equity_last = record.get('trade').get('equity')
    record.get('trade').update({'equity': round((position * new_price + money)/10000, 3)})
    equity = record.get('trade').get('equity')
    # equity_max
    record.get('trade').update({'equity_max': max(record.get('trade').get('equity_max'), equity)})
    # return_r, by yealy
    return_r = round(equity - 1, 4)
    days = record.get('key').get('count_all')
    record.get('trade').update({'return_r': round(np.log(1 + return_r) / days * 252, 3)})
    # draw_r
    equity_max = record.get('trade').get('equity_max')
    draw_r = round(1 - equity / equity_max, 3)
    record.get('trade').update({'draw_r': draw_r})
    # max_draw_r
    record.get('trade').update({'max_draw_r': max(record.get('trade').get('max_draw_r'), draw_r)})
    # sharp_r
    return_r_daily = record.get('sig').get('return_r_daily')
    return_r_daily.append(round(equity / equity_last - 1, 3))
    record.get('sig').update({'return_r_daily': return_r_daily})
    no_risk_r = round(np.e ** (0.03 / 365) - 1, 4)
    if np.std(return_r_daily):
        sharp_r = round((np.mean(return_r_daily) - no_risk_r) / np.std(return_r_daily) * np.sqrt(252), 3)
    else:
        sharp_r = np.nan
    record.get('trade').update({'sharp_r': sharp_r})

    return None


def record_update(sample, record):
    path_out = sample.replace('.', '_log.')
    pd.DataFrame().to_csv(path_out, encoding='utf-8-sig', index=False, header=False)
    path_tidy = sample.replace('.', '_tidy.')
    reader = pd.read_csv(path_tidy, usecols=['date', 'new_price', 's1', 's2', 's3'], chunksize=5000)
    result = []
    cols = [
        'date',
        'new_price',
        'col_name',
        'col_price',
        'signal',
        'buy_price',
        'sell_price',
        'equity',
        'draw_r',
        'return_r',
        'max_draw_r',
        'sharp_r',
    ]
    for chunk in reader:
        record.get('sig').update({'result': []})
        # record.get('sig').update({'return_r_daily': []})
        for i in np.arange(chunk.shape[0]):
            new_price = chunk.iloc[i, 1].round(3)
            s1 = chunk.iloc[i, 2].round(3)
            s2 = chunk.iloc[i, 3].round(3)
            s3 = chunk.iloc[i, 4].round(3)
            match_rule(new_price, record, s1, s2, s3)
            result = record.get('sig').get('result')
            result.append([
                chunk.iloc[i, 0],
                new_price,
                record.get('last').get('column'),
                record.get('last').get('new_price'),
                record.get('sig').get('signal'),
                record.get('trade').get('buy_price_in'),
                record.get('trade').get('buy_price_out'),
                record.get('trade').get('equity'),
                record.get('trade').get('draw_r'),
                record.get('trade').get('return_r'),
                record.get('trade').get('max_draw_r'),
                record.get('trade').get('sharp_r'),
            ])
            record.get('sig').update({'result': result})
        print('/', end='')
        pd.DataFrame(result, columns=cols).set_index('date').to_csv(path_out, mode='a', encoding='utf-8-sig')
    print(record.get('sig').get('signal'))
    # with open('update_log.txt', mode='w', encoding='utf-8-sig') as f:
    #     for k, v in record.items():
    #         f.write('{}: {}\n'.format(k, v))

    return None


def back_result(sample, train_or_test, p1, p2, p3, p4):
    path_log = sample.replace('.', '_log.')
    path_tidy = sample.replace('.', '_tidy.')
    result = pd.read_csv(path_log)
    result = pd.merge(result, pd.read_csv(path_tidy))
    result.drop_duplicates(inplace=True)
    result.reset_index(inplace=True)
    result.set_index('date', inplace=True)
    result.sort_index(inplace=True)
    result['buy_p'] = result['buy_price'].ffill()
    result['buy_r'] = round(result['sell_price'] / result['buy_p'] * 100 - 100, 3)

    # Index(['index', 'new_price', 'col_name', 'col_price', 'signal', 'buy_price',
    #        'sell_price', 'equity', 'draw_r', 'return_r', 'max_draw_r', 'sharp_r'],
    #       dtype='object')

    path_log = path_log.replace('_log.', '_{}_{}_{}_{}_{}.'.format(p1, p2, p3, p4, train_or_test))
    # (1) price
    title = path_log.replace('.csv', '_plot_price.png')
    fig, axes = plt.subplots(figsize=(80, 40))
    result['new_price'].plot(marker='o', color='grey', legend=True, ax=axes, alpha=0.5)
    result['buy_price'].plot(marker='^', color='r', legend=True, ax=axes)
    result['sell_price'].plot(marker='v', color='g', legend=True, ax=axes, grid=True)

    t_list = [
        ['b', 'buy_price', 'r', 1],
        ['s', 'sell_price', 'g', 1.5, 'r', 'buy_r', 2, 'grey'],
    ]
    for i in t_list:
        for j in np.arange(result[i[1]].shape[0]):
            if result[i[1]][j] > 0:
                axes.annotate(
                    '{}:{}'.format(i[0], result[i[1]][j].round(2)),
                    xy=(j, result[i[1]][j]),
                    xytext=(j, result[i[1]][j] - i[3]),
                    fontsize=16,
                    color='{}'.format(i[2]),
                )
                if i[0] == 's':
                    axes.annotate(
                        '{}:{}%'.format(i[4], result[i[5]][j].round(2)),
                        xy=(j, result[i[1]][j]),
                        xytext=(j, result[i[1]][j] - i[6]),
                        fontsize=16,
                        color='{}'.format(i[7]),
                    )

    axes.set_title(title)
    plt.savefig(title)
    # plt.show()
    plt.close(fig)

    # (2) equity
    result['draw_r'] = -result['draw_r'].replace(0, np.nan)
    title = path_log.replace('.csv', '_plot_equity.png')
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    # result['equity'].plot(color='orange', legend=True, ax=axes)
    # result['draw_r'].plot(color='grey', legend=True, ax=axes)

    result['equity'].plot(color='orange', legend=True, ax=axes[0])
    result['draw_r'].plot(marker='o', color='silver', legend=True, ax=axes[1])

    # t_list = [
    #     ['md', 'draw_r', 'head'],
    #     ['low', 'equity', 'head'],
    #     ['max', 'equity', 'tail'],
    #     ['sr', 'sharp_r', 'sort_index().tail'],
    #     ['rr', 'return_r', 'sort_index().tail'],
    # ]
    # for i in t_list:
    #     i_index = eval('result[i[1]].sort_values().'+i[2]+'(1).index')
    #     axes.annotate(
    #         '{}: {}'.format(i[0], result.loc[i_index, i[1]].round(3)[0]),
    #         xy=(result.loc[i_index, 'index'], result.loc[i_index, i[1]]),
    #         xytext=(result.loc[i_index, 'index'], result.loc[i_index, i[1]] - 0.05),
    #         fontsize=16,
    #         color='b',
    #     )

    t_list = [
        ['low', 'equity', 'head'],
        ['max', 'equity', 'tail'],
        ['sr', 'sharp_r', 'sort_index().tail'],
    ]
    for i in t_list:
        i_index = eval('result[i[1]].sort_values().'+i[2]+'(1).index')
        axes[0].annotate(
            '{}: {}'.format(i[0], result.loc[i_index, i[1]].round(3)[0]),
            xy=(result.loc[i_index, 'index'], result.loc[i_index, i[1]]),
            xytext=(result.loc[i_index, 'index'], result.loc[i_index, i[1]] * 1.1),
            fontsize=16,
            color='b',
        )

    t_list = [
        ['md', 'draw_r', 'head', 'draw_r'],
        ['rr', 'return_r', 'sort_index().tail', 'draw_r'],
    ]
    for i in t_list:
        i_index = eval('result[i[1]].sort_values().'+i[2]+'(1).index')
        axes[1].annotate(
            '{}: {}%'.format(i[0], round(result.loc[i_index, i[1]][0]*100), 3),
            xy=(result.loc[i_index, 'index'], result.loc[i_index, i[3]]),
            xytext=(result.loc[i_index, 'index'], result.loc[i_index, i[3]] * 0.95),
            fontsize=16,
            color='b',
        )

    axes[0].fill_between(result['index'], 1, result['equity'], where=result['equity'] <= 1, facecolor='yellowgreen', alpha=0.5)
    axes[1].fill_between(result['index'], 0, result['draw_r'], where=result['draw_r'] <= 0, facecolor='silver', alpha=0.5)

    axes[0].set_title(title)
    plt.savefig(title)
    # plt.show()
    plt.close(fig)

    # (3) paras
    title = path_log.replace('.csv', '_plot_paras.png')
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    result.loc[:, ['s1', 's2', 's3']].plot(legend=True, grid=True, ax=axes)

    axes.set_title(title)
    plt.savefig(title)
    # plt.show()
    plt.close(fig)

    return None


def split_sample(sample, p5, p6):
    df_price = pd.read_csv(sample, encoding='utf-8-sig', parse_dates=[0], skiprows=p5)
    df_price.drop_duplicates(keep='last', inplace=True)
    df_price.columns = ['date', 'new_price']
    df_price['new_price'] = df_price['new_price'].round(3)
    path_train = sample.replace('.', '__train.')
    path_test = sample.replace('.', '__test.')
    df_price.loc[:p6, ['date', 'new_price']].to_csv(path_train, encoding='utf-8-sig')
    df_price.loc[p6:, ['date', 'new_price']].to_csv(path_test, encoding='utf-8-sig')

    return None


def scale_tidy(sample, p1, p2, p3, p4, mode, train_or_test):
    path_train = sample.replace('.', '__{}.'.format(train_or_test))
    df_price = pd.read_csv(path_train, encoding='utf-8-sig', parse_dates=[0], index_col=[0])
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
    df_price.loc[:, ['new_price', 's1', 's2', 's3']].to_csv(path_tidy, encoding='utf-8-sig')

    return None


if __name__ == '__main__':
    st = time.time()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # date, new_price, sample: xx.csv
    sample_pool = [
        '601519.csv',
    ]

    # p1, p2, p3, p4: windows,
    # p5: skip rows,
    # p6: train rows,
    para_pool = {
        'train_or_test': ['train', 'test'],
        'mode': [6],
        'p1': [0.2],
        'p2': [0.1],
        'p3': [0.05],
        'p4': [51],
        'p5': [0],
        'p6': [252 * 4.59],
    }

    for sample in sample_pool:
        for train_or_test in para_pool.get('train_or_test'):
            for p5 in para_pool.get('p5'):
                for p6 in para_pool.get('p6'):
                    p5 = np.round(p5, 3)
                    p6 = np.round(p6, 3)
                    split_sample(sample, p5, p6)

            for p1 in para_pool.get('p1'):
                for p2 in para_pool.get('p2'):
                    for p3 in para_pool.get('p3'):
                        for p4 in para_pool.get('p4'):
                            # xx_tidy.csv
                            scale_tidy(sample, p1, p2, p3, p4, mode=para_pool.get('mode')[0], train_or_test=train_or_test)
                            # xx_log.csv
                            record_update(sample, record_init())
                            # xx_log_plot_xx.png
                            back_result(sample, train_or_test, p1, p2, p3, p4)

    print(time.time() - st)
