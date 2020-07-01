#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/12/31 19:40

@author: tatatingting
"""
from unittest import result

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


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
    }})

    record.update({'trade': {
        'buy_in_day': False,
        'buy_out_day': False,
        'buy_in': False,
        'buy_price_in': np.nan,
        'buy_price_out': np.nan,
        'c_succ': 0,
        'r_win': 0,
        'c_fail': 0,
        'r_lose': 0,
        'e': 0,
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

    # buy-in
    if record.get('trade').get('buy_in_day'):
        record.get('trade').update({'buy_in_day': False})
        record.get('trade').update({'buy_in': True})
        record.get('trade').update({'buy_price_in': new_price})

    if record.get('trade').get('buy_out_day'):
        record.get('trade').update({'buy_out_day': False})
        record.get('trade').update({'buy_in': False})
        record.get('trade').update({'buy_price_out': new_price})
        buy_price = record.get('trade').get('buy_price_in')
        r = np.around(new_price / buy_price - 1, 5)
        if r > 0:
            c_succ = record.get('trade').get('c_succ')
            r_win = record.get('trade').get('r_win')
            record.get('trade').update({'c_succ': np.int(c_succ + 1)})
            record.get('trade').update({'r_win': np.around(r_win + r, 5)})
        else:
            c_fail = record.get('trade').get('c_fail')
            r_lose = record.get('trade').get('r_lose')
            record.get('trade').update({'c_fail': np.int(c_fail + 1)})
            record.get('trade').update({'r_lose': np.around(r_lose + r, 5)})

    buy_in = record.get('trade').get('buy_in')
    signal = record.get('sig').get('signal')

    # in
    if not buy_in and ('buy_near' in signal):
        record.get('trade').update({'buy_in_day': True})
    if buy_in and (('indicate_up_over' in signal) or ('sell_near' in signal) or ('positively_down_resumed' in signal)):
        record.get('trade').update({'buy_out_day': True})

    return None


def split_sample(sample, p5, p6):
    df_price = pd.read_csv(sample, encoding='utf-8-sig', parse_dates=[0], skiprows=p5)

    df_price.drop_duplicates(keep='last', inplace=True)
    df_price.columns = ['date', 'new_price']
    df_price.loc[:, 'new_price'] = df_price['new_price'].round(3)
    path_train = sample.replace('.', '__train.')
    path_test = sample.replace('.', '__test.')
    df_price.loc[:p6, ['date', 'new_price']].to_csv(path_train, encoding='utf-8-sig', index=False)
    df_price.loc[p6:, ['date', 'new_price']].to_csv(path_test, encoding='utf-8-sig', index=False)

    return None


def get_price(path_test_or_train):
    df_price = pd.read_csv(path_test_or_train, encoding='utf-8-sig', parse_dates=[0])
    df_price.drop_duplicates(keep='last', inplace=True)
    df_price.columns = ['date', 'new_price']
    df_price.sort_values(by='date', inplace=True)
    df_price['new_price'] = df_price['new_price'].round(3)

    return df_price


def tidy_chunk(chunk_price, mode, p1, p2, p3, p4):
    chunk_price.drop_duplicates(keep='last', inplace=True)
    chunk_price.columns = ['date', 'new_price']
    chunk_price.set_index(keys='date', inplace=True)
    chunk_price.sort_index(inplace=True)
    chunk_price['new_price'] = chunk_price['new_price'].round(3)

    chunk_price['1_diff_mean'] = chunk_price['new_price'].diff().rolling(window=p4).mean().abs().round(3)
    chunk_price['1_diff_std'] = chunk_price['new_price'].diff().rolling(window=p4).std().round(3)

    chunk_price['2_diff_mean'] = chunk_price['new_price'].diff().abs().rolling(window=p4).mean().round(3)
    chunk_price['2_diff_std'] = chunk_price['new_price'].diff().abs().rolling(window=p4).std().round(3)

    chunk_price['new_price_max'] = chunk_price['new_price'].rolling(window=p4).max().round(3)
    chunk_price['new_price_min'] = chunk_price['new_price'].rolling(window=p4).min().round(3)
    chunk_price['new_price_max_min'] = chunk_price['new_price_max'] - chunk_price['new_price_min']

    chunk_price['new_price_mean'] = chunk_price['new_price'].rolling(window=p4).mean().round(3)

    chunk_price['2_diff_p'] = chunk_price['2_diff_mean'] / chunk_price['new_price']

    chunk_price.fillna(method='bfill', inplace=True)

    if mode == 1:
        chunk_price['s1'] = chunk_price['1_diff_mean'] + chunk_price['1_diff_std'] * p1
        chunk_price['s2'] = chunk_price['1_diff_mean'] + chunk_price['1_diff_std'] * p1 * 0.5
        chunk_price['s3'] = chunk_price['1_diff_mean'] + chunk_price['1_diff_std'] * p3

    elif mode == 2:
        chunk_price['s1'] = chunk_price['2_diff_mean'] + chunk_price['2_diff_std'] * p1
        chunk_price['s2'] = chunk_price['2_diff_mean'] + chunk_price['2_diff_std'] * p2
        chunk_price['s3'] = chunk_price['2_diff_mean'] + chunk_price['2_diff_std'] * p3

    elif mode == 3:
        chunk_price['s1'] = chunk_price['2_diff_mean'] + chunk_price['2_diff_std'] * p1
        chunk_price['s2'] = chunk_price['2_diff_mean'] + chunk_price['2_diff_std'] * p2
        chunk_price['s3'] = chunk_price['1_diff_mean'] + chunk_price['1_diff_std'] * p3

    elif mode == 4:
        chunk_price['s1'] = chunk_price['2_diff_mean'] * p1
        chunk_price['s2'] = chunk_price['2_diff_mean'] * p2
        chunk_price['s3'] = chunk_price['2_diff_mean'] * p3

    elif mode == 5:
        chunk_price['s1'] = chunk_price['1_diff_mean'] * p1
        chunk_price['s2'] = chunk_price['1_diff_mean'] * p2
        chunk_price['s3'] = chunk_price['1_diff_mean'] * p3

    elif mode == 6:
        chunk_price['s1'] = chunk_price['new_price_max_min'] * p1
        chunk_price['s2'] = chunk_price['s1'] * p2
        chunk_price['s3'] = chunk_price['new_price'] * p3

    elif mode == 7:
        chunk_price['s1'] = chunk_price['new_price_mean'] * p1
        chunk_price['s2'] = chunk_price['new_price_mean'] * p1 * 0.5
        chunk_price['s3'] = chunk_price['new_price_mean'] * p3

    elif mode == 8:
        chunk_price['s1'] = chunk_price['new_price_max_min'] * p1
        chunk_price['s2'] = chunk_price['new_price_max_min'] * p1 * 0.5
        chunk_price['s3'] = chunk_price['2_diff_p'] * p3

    chunk = chunk_price.loc[:, ['new_price', 's1', 's2', 's3']]

    return chunk


def run_find(sample, paras):
    p5 = np.int(paras.get('p5')[0])
    p6 = np.int(paras.get('p6')[0])
    p8 = np.int(paras.get('p8')[0])

    split_sample(sample, p5, p6)

    path_result = sample.replace('.', '_out_{}.'.format(np.int(time.time())))
    cols = ['p1', 'p2', 'p3', 'p4', 'train_or_test', 'count_trade', 'c_succ', 'c_fail', 'r_win', 'r_lose', 'e_value']
    pd.DataFrame(columns=cols).to_csv(path_result, encoding='utf-8-sig', mode='w', index=False)
    res = []

    for train_or_test in paras.get('train_or_test'):
        # get chunk set
        path_test_or_train = sample.replace('.', '__{}.'.format(train_or_test))
        df_price = get_price(path_test_or_train)

        for p1 in paras.get('p1'):
            for p2 in paras.get('p2'):
                for p3 in paras.get('p3'):
                    for p4 in paras.get('p4'):

                        p1 = np.round(p1, 3)
                        p2 = np.round(p2, 3)
                        p3 = np.round(p3, 3)
                        p4 = np.int(p4)

                        count_trade = 0
                        c_succ = 0
                        c_fail = 0
                        r_win = 0
                        r_lose = 0
                        e_value = 0
                        n_count_trade = paras.get('n_count_trade')[0]
                        b = 0
                        while count_trade < n_count_trade:
                            a = count_trade

                            # get the chunk
                            start_index = np.random.randint(0, df_price.shape[0] - p8)
                            chunk_price = df_price.iloc[start_index:(start_index + p8), :]

                            # tidy the chunk
                            mode = np.int(paras.get('mode')[0])
                            chunk = tidy_chunk(chunk_price.copy(), mode, p1, p2, p3, p4)

                            # sim trade
                            record = record_init()
                            for i in np.arange(p8):
                                new_price = chunk.iloc[i, 0].round(3)
                                s1 = chunk.iloc[i, 1].round(3)
                                s2 = chunk.iloc[i, 2].round(3)
                                s3 = chunk.iloc[i, 3].round(3)
                                match_rule(new_price, record, s1, s2, s3)

                            c_succ += record.get('trade').get('c_succ')
                            r_win += record.get('trade').get('r_win')
                            c_fail += record.get('trade').get('c_fail')
                            r_lose += record.get('trade').get('r_lose')
                            count_trade += record.get('trade').get('c_succ')
                            count_trade += record.get('trade').get('c_fail')
                            if count_trade == a:
                                b += 1
                            if b > 5:
                                break

                        # result out
                        if (c_succ + c_fail) > 0:
                            e_value = np.around((r_win + r_lose) / (c_succ + c_fail), 3)
                        res.append([
                            p1,
                            p2,
                            p3,
                            p4,
                            train_or_test,
                            count_trade,
                            c_succ,
                            c_fail,
                            round(r_win, 3),
                            round(r_lose, 3),
                            e_value,
                        ])
                        print('/', end='')

    df_res = pd.DataFrame(res)
    df_res.to_csv(path_result, encoding='utf-8-sig', mode='a', index=False, header=False)
    df_res.drop_duplicates(inplace=True)
    df_res.columns = cols
    df_res['succ_fail'] = np.around(df_res['c_succ'] / df_res['count_trade'], 3)
    df_res['win_lose'] = np.around(abs(df_res['r_win'] / df_res['r_lose']), 3)
    path_out = pd.ExcelWriter(path_result.replace('.csv', '_pivot.xlsx'))
    for col in ['count_trade', 'succ_fail', 'win_lose', 'e_value']:
        df_res_pivot = df_res.pivot_table(index=['train_or_test', 'p1'], columns=['p4'], values=[col])
        df_res_pivot.to_excel(path_out, encoding='utf-8-sig', sheet_name=col)
    path_out.save()

    return None


if __name__ == '__main__':
    para_pool = {
        'n_count_trade': [30],
        'train_or_test': ['train', 'test'],
        'mode': [7],
        'p1': np.arange(0, 1, 0.05),
        'p2': [],
        'p3': [0.05],
        'p4': np.arange(5, 70, 5),
        'p5': [0],
        'p6': [252 * 4.59],
        'p7': np.nan,
        'p8': [252],
    }

    run_find(sample='601519.csv', paras=para_pool)
