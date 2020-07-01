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


# feiqi de
def run_find_random(samples_count, samples_days, paras):
    sample = '602019_random.csv'
    path_out = sample.replace('.', '_find_para_{}.'.format(time.time()))
    with open(path_out, mode='w', encoding='utf-8-sig') as f:
        f.write('i,walks_jizhi,walks_max,walks_min,sample,p1,p2,p3,p4,p5,return_r,max_draw_r,sharp_r\n')

    nwalks = samples_count
    nsteps = samples_days
    draws = np.random.randint(0, 2, size=(nwalks, nsteps))  # 0, 1
    steps = np.where(draws > 0, 1, -1)
    walks = steps.cumsum(1)
    walks_min_list = walks.min(axis=1)
    walks = walks - np.tile(walks_min_list, (nsteps, 1)).T

    for i in np.arange(walks.shape[0]):
        sample = '602019_random_{}.csv'.format(i)
        walks[i] += np.random.randint(1, 100)
        walks_max = walks[i].max()
        walks_min = walks[i].min()
        walks_jizhi = walks_max - walks_min

        for p1 in paras.get('p1'):
            for p2 in paras.get('p2'):
                for p3 in paras.get('p3'):
                    for p4 in paras.get('p4'):
                        for p5 in paras.get('p5'):
                            p1 = np.round(p1, 3)
                            p2 = np.round(p2, 3)
                            p3 = np.round(p3, 3)
                            p4 = np.round(p4, 3)
                            p5 = np.round(p5, 3)
                            pd.DataFrame({'date': pd.date_range(start='1/1/2018', periods=nsteps),
                                          'new_price': walks[i]}).to_csv(sample, index=False)
                            scale_tidy(sample, p1, p2, p3, p4)
                            record = record_init()
                            find_record_update(sample, record)
                            with open(path_out, mode='a', encoding='utf-8-sig') as f:
                                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                                    i, walks_jizhi, walks_max, walks_min, sample, p1, p2, p3, p4, p5,
                                    record.get('trade').get('return_r'),
                                    record.get('trade').get('max_draw_r'),
                                    record.get('trade').get('sharp_r'),
                                ))
        # print('\n', path_out, i)
    return None


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
    record.get('trade').update({'equity': round((position * new_price + money) / 10000, 3)})
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


def back_result_3d(path, paras, cols):
    df = pd.read_csv(path)
    df.reset_index(drop=True, inplace=True)

    # diy nots of 3d plotting
    df.sort_values(by=eval(cols), ascending=True, inplace=True)

    x = paras.get('p1')
    y = paras.get('p2')
    z = paras.get('p3')

    X, Y, Z = np.meshgrid(y, x, z)  # yes, that's it: y, x, z crazy!!!
    print(df.shape, X.shape, Y.shape, Z.shape)

    for metric in ['return_r', 'max_draw_r', 'sharp_r']:
        c = df.groupby(by=eval(cols))[metric].mean() * 1
        C = np.array(c)
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.set_xlabel(xlabel=eval(cols)[1])
        ax1.set_ylabel(ylabel=eval(cols)[0])
        ax1.set_zlabel(zlabel=eval(cols)[2])
        ax1.scatter(X, Y, Z, c=C, cmap=plt.get_cmap('coolwarm'))
        plt.legend(metric)
        plt.show()


def back_result_2d(path, pp, pnvalue):
    df = pd.read_csv(path, usecols=pp + ['return_r', 'max_draw_r', 'sharp_r'])
    df.reset_index(drop=True, inplace=True)
    df = df.loc[df[pp[2]] == pnvalue, [pp[0], pp[1], 'return_r', 'max_draw_r', 'sharp_r']]
    # excel
    df.sort_values(by=[pp[0], pp[1]], ascending=True, inplace=True)
    path_out = pd.ExcelWriter(path + '_out.xlsx')
    for col in df.columns[-3:]:
        pd.pivot_table(df, values=col, index=pp[0], columns=pp[1], aggfunc=np.mean).to_excel(path_out,
                                                                                             sheet_name=col)
    path_out.save()

    return None


def back_result_2d_start_index(path, pp):
    df_start = pd.read_csv(path)

    df_start['sr'] = df_start['sharp_r']
    df_start['rr'] = df_start['return_r']
    df_start['md'] = df_start['max_draw_r']
    cols = ['sr', 'rr', 'md']

    # excel
    path_out = pd.ExcelWriter(path + '_out_2d_start_index.xlsx')
    df_all = df_start.pivot_table(values=cols, index=pp, columns='start_index', aggfunc=np.mean)
    df_all_scale = pd.DataFrame()
    for col in cols:
        df = df_all[col]
        df.to_excel(path_out, sheet_name=col+'_data')

        # df_mean = df.mean(axis=1).reset_index().pivot(index='p1', columns='p2')
        # df_mean.to_excel(path_out, sheet_name=col+'_mean')
        #
        # df_std =  df.std(axis=1).reset_index().pivot(index='p1', columns='p2')
        # df_mean.to_excel(path_out, sheet_name=col+'_std')
        #
        # df_mean_chu_std = df_mean / df_std
        # df_mean_chu_std.to_excel(path_out, sheet_name=col+'_mean_chu_std')

        # df_mean = pd.DataFrame(np.tile(df.mean(axis=1), (9, 1)).T)
        # df_std = pd.DataFrame(np.tile(df.std(axis=1), (9, 1)).T)
        # df_scale = (df - np.array(df_mean)) / np.array(df_std)
        # df_scale.to_excel(path_out, sheet_name=col + '_scale')

        df_mean = df.mean(axis=1)
        df_mean.name = col
        df_all_scale = pd.concat((df_all_scale, df_mean), axis=1)

        df_mean_std = (df.mean(axis=1)/df.std(axis=1))
        df_mean_std.name = col + '_ms'
        # df_mean_std.to_excel(path_out, sheet_name=col + '_mean_std')
        df_all_scale = pd.concat((df_all_scale, df_mean_std), axis=1)

    sr_count = df_all['sr'].count(axis=1)
    sr_count.name = 'sr_count'
    df_all_scale = pd.concat((df_all_scale, sr_count), axis=1)

    rr_count = df_all['rr'].count(axis=1)
    # rr_0_count = pd.DataFrame(np.where(df_all['rr'] > 0, 1, 0).sum(axis=1))
    rr_0_count = (df_all['rr'] > 0).sum(axis=1)
    df_rr_r = rr_0_count / rr_count
    df_rr_r.name = '_win'
    df_all_scale = pd.concat((df_all_scale, df_rr_r), axis=1)

    md_count = df_all['md'].count(axis=1)
    md_30_count = (df_all['rr'] < 0.29).sum(axis=1)
    df_md_r = md_30_count / md_count
    df_md_r.name = '_md30'
    df_all_scale = pd.concat((df_all_scale, df_md_r), axis=1)

    rr_count = df_all['rr'].count(axis=1)
    rr_30_count = (df_all['rr'] > 0.29).sum(axis=1)
    df_rr_r = rr_30_count / rr_count
    df_rr_r.name = '_win30'
    df_all_scale = pd.concat((df_all_scale, df_rr_r), axis=1)


    df_all_scale['rsm'] = df_all_scale['rr'] * abs(df_all_scale['sr']) / df_all_scale['md']

    df_all_scale.sort_values(by=['_win', 'sr_count', '_md30', '_win30', 'rsm'], ascending=False).to_excel(path_out, sheet_name='all_metric')
    path_out.save()

    df_all_scale.sort_values(by=['_win', 'sr_count', '_md30', '_win30', 'rsm'], ascending=False).head(100).to_csv('result.csv')

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


def run_find(sample, paras):
    p5 = np.int(paras.get('p5')[0])
    p6 = np.int(paras.get('p6')[0])
    p7 = np.int(paras.get('p7')[0])
    p8 = np.int(paras.get('p8')[0])
    result_lines = []
    # path_out = sample.replace('.', '_find_para_{}.'.format(str(time.time()).replace('.', '')))
    # path_out = sample.replace('.', '_find_para.')
    path_out = sample.replace('.', '_find_para_{}.'.format(np.random.randint(10)))
    cols = ['sample', 'mode', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'start_index', 'return_r', 'max_draw_r', 'sharp_r']
    pd.DataFrame(columns=cols).to_csv(path_out, index=False)
    for train_loop in np.arange(p7):
        print(train_loop)
        start_index = np.random.randint(0, p6 - p8)
        split_sample(sample, p5, p6)
        for p1 in paras.get('p1'):
            for p2 in paras.get('p2'):
                for p3 in paras.get('p3'):
                    for p4 in paras.get('p4'):
                        p1 = np.round(p1, 3)
                        p2 = np.round(p2, 3)
                        p3 = np.round(p3, 3)
                        p4 = np.int(p4)
                        path_train = sample.replace('.', '__{}.'.format(paras.get('train_or_test')))
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

                        df_price['2_diff_p'] = df_price['2_diff_mean'] / df_price['new_price']

                        df_price.fillna(method='bfill', inplace=True)

                        mode = np.int(paras.get('mode')[0])
                        if mode == 1:
                            df_price['s1'] = df_price['1_diff_mean'] + df_price['1_diff_std'] * p1
                            df_price['s2'] = df_price['1_diff_mean'] + df_price['1_diff_std'] * p1 * 0.5
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
                        elif mode == 8:
                            df_price['s1'] = df_price['new_price_max_min'] * p1
                            df_price['s2'] = df_price['s1'] * p2
                            df_price['s3'] = df_price['2_diff_p'] * p3

                        chunk = df_price.loc[:, ['new_price', 's1', 's2', 's3']]
                        # path_tidy = sample.replace('.', '_tidy.')
                        # chunk.to_csv(path_tidy, encoding='utf-8-sig')
                        chunk = chunk.iloc[start_index:(start_index + p8), :]

                        record = record_init()

                        for i in np.arange(chunk.shape[0]):
                            new_price = chunk.iloc[i, 0].round(3)
                            s1 = chunk.iloc[i, 1].round(3)
                            s2 = chunk.iloc[i, 2].round(3)
                            s3 = chunk.iloc[i, 3].round(3)
                            match_rule(new_price, record, s1, s2, s3)
                        print('/', end='')

                        result_line = [
                            sample, mode, p1, p2, p3, p4, p5, p6, p7, p8, start_index,
                            record.get('trade').get('return_r'),
                            record.get('trade').get('max_draw_r'),
                            record.get('trade').get('sharp_r'),
                        ]
                        result_lines.append(result_line)

            pd.DataFrame(result_lines).to_csv(path_out, mode='a', index=False, header=False)
            result_lines = []
        print('/', end='\n')
    print('\n', path_out)
    return None


if __name__ == '__main__':
    st = time.time()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # mode, p1, p2, p3,
    # p4: windows,
    # p5: skip rows,
    # p6: train rows,
    # p7: train loops,
    # p8: batch for each train loop,
    # 2166/8.59
    para_pool_find = {
        'train_or_test': 'train',
        'mode': [1],
        'p1': np.arange(0, 2, 0.2),
        'p2': [0],
        'p3': np.arange(0, 0.1, 0.05),
        'p4': np.arange(1, 72, 10),
        'p5': [0],
        'p6': [252 * 4.59],
        'p7': [30],
        'p8': [252],
    }
    sample_pool = ['601519.csv']

    # run_find(sample_pool[0], para_pool_find)
    back_result_2d_start_index('601519_find_para_8.csv', pp=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'])
    # back_result_2d('601519_find_para_?.csv', pp=['p1', 'p2', 'p3'], pnvalue=0.01)
    # back_result_3d('601519_find_para.csv', para_pool_find, "['p1', 'p2', 'p3']")

    print(time.time() - st)
