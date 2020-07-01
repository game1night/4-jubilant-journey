#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/11/22 9:26

@author: tatatingting
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# from scipy import stats


# 历史记录的初始化
def init(start='begin'):
    record = {}
    for i in ['ciup', 'ziup', 'up', 'down', 'zidown', 'cidown']:
        record.update({i: {}})

    for i in ['ziup', 'down', 'ciup']:
        record.update({i: {'column': i, 'new_price': float('-Inf')}})
    for i in ['up', 'zidown', 'cidown']:
        record.update({i: {'column': i, 'new_price': float('Inf')}})

    for i in ['trade', 'sig', 'key', 'last']:
        record.update({i: {}})

    record.get('key').update({
        'count_all': 0,
        'count_at': 0,
        'point_up': float('Inf'),
        'point_down': float('-Inf'),
        'point_down_from': 'zidown',
        'point_up_from': 'ziup',
        'buy_point': float('-Inf'),
        'sell_point': float('Inf'),
        'judge_up': float('Inf'),
        'judge_down': float('-Inf'),
        'max_up': float('-Inf'),
        'min_down': float('Inf'),
    })

    record.get('trade').update({
        'buy_in': False,
        'buy_in_day': False,
        'buy_out_day': False,
        'buy_price_in': 0,
        'buy_price_out': 0,
        'count_trade': 0,
        'count_succ': 0,
        'profit': 10000,
        'money': 10000,
        'position': 0,
        'profit_max': 10000,
        'profit_p': 1,
        'draw_p': 0,
        'profit_p_daily': 0,
        'buy_profit_p': 0,
    })

    # print('init done')

    if start == 'book':
        # 加载模板（原本）

        new_column = 'down'
        new_price = 48.25
        record.get('last').update({'column': new_column, 'new_price': new_price, 'time': time.time()})
        record.get(new_column).update({'column': new_column, 'new_price': new_price, 'time': time.time()})

        record.get('key').update({'point_up': 62.125, 'point_up_from': 'ziup'})
        record.get('key').update({'point_down': 48.5, 'point_down_from': 'down'})

        # print('load data from book done')

    if start == 'begin':
        new_column = 'up'
        new_price = 0
        record.get('last').update({'column': new_column, 'new_price': new_price, 'time': time.time()})
        record.get(new_column).update({'column': new_column, 'new_price': new_price, 'time': time.time()})

        # print('load data of begin done')

    return record


# 匹配规则
def match_rule(new_price, record, s1, s2, s3):
    record.get('key').update({'s1': s1, 's2': s2, 's3': s3})

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
        record.get('key').update({'count_at': record.get('key').get('count_at') + 1})
        record.get('last').update({'column': new_column, 'new_price': new_price, 'time': time.time()})
        record.get(new_column).update({'column': new_column, 'new_price': new_price, 'time': time.time()})

    # 匹配信号

    signal = []

    last_column = record.get('last').get('column')
    buy_point = record.get('key').get('buy_point')
    sell_point = record.get('key').get('sell_point')
    judge_up = record.get('key').get('judge_up')
    judge_down = record.get('key').get('judge_down')

    # whether_to_up: ~P
    # whether_to_down

    # sell_near: ~p
    # positively_up_resumed: >p+3
    # indicate_up_over: <max & <p-3
    # danger_indicate_up_over: <max-3 & max<~P

    # buy_near
    # positively_down_resumed
    # indicate_down_over
    # danger_indicate_down_over

    if last_column in ['up', 'ziup', 'ciup']:
        record.get('key').update({'trend': 'up'})
    else:
        record.get('key').update({'trend': 'down'})
    trend = record.get('key').get('trend')
    max_up = record.get('key').get('max_up')
    min_down = record.get('key').get('min_down')
    if trend == 'up':
        record.get('key').update({'max_up': max(new_price, max_up)})
        record.get('key').update({'min_down': float('Inf')})
    else:
        record.get('key').update({'min_down': min(new_price, min_down)})
        record.get('key').update({'max_up': float('-Inf')})

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

    return None


# 模拟交易
def sim_trade(new_price, record, pic):
    # buy-in
    if record.get('trade').get('buy_in_day'):
        record.get('trade').update({'buy_in_day': False})
        record.get('trade').update({'buy_in': True})
        record.get('trade').update({'buy_price_in': new_price})
        record.get('trade').update({'position': round(record.get('trade').get('money') / new_price, 5)})
        record.get('trade').update({'money': 0})
        if pic:
            print('{}, {}'.format(record.get('key').get('count_all'),
                                  record.get('trade').get('buy_price_in')))
    if record.get('trade').get('buy_out_day'):
        record.get('trade').update({'buy_out_day': False})
        record.get('trade').update({'buy_in': False})
        record.get('trade').update({'count_trade': record.get('trade').get('count_trade') + 1})
        record.get('trade').update({'buy_price_out': new_price})
        record.get('trade').update({'money': round(record.get('trade').get('position') * new_price, 5)})
        record.get('trade').update({'position': 0})
        buy_price_in = record.get('trade').get('buy_price_in')
        buy_price_out = record.get('trade').get('buy_price_out')
        record.get('trade').update({'buy_profit_p': round(buy_price_out / buy_price_in * 100 - 100, 5)})
        if pic:
            print('{}, {} ---( {}%)'.format(record.get('key').get('count_all'),
                                            buy_price_out,
                                            record.get('trade').get('buy_profit_p')))
        if buy_price_in < buy_price_out:
            record.get('trade').update({'count_succ': record.get('trade').get('count_succ') + 1})

    signal = record.get('sig').get('signal')
    buy_in = record.get('trade').get('buy_in')

    # in
    if not buy_in and ('buy_near' in signal):
        record.get('trade').update({'buy_in_day': True})
    if buy_in and (('indicate_up_over' in signal) or ('sell_near' in signal) or ('positively_down_resumed' in signal)):
        record.get('trade').update({'buy_out_day': True})

    return None


# 计算交易
def calculate_trade(new_price, record):
    position = record.get('trade').get('position')
    money = record.get('trade').get('money')

    # profit
    profit_last = record.get('trade').get('profit')
    record.get('trade').update({'profit': round(position * new_price + money, 5)})
    profit = record.get('trade').get('profit')
    # profit_max
    profit_max = record.get('trade').get('profit_max')
    record.get('trade').update({'profit_max': max(profit_max, profit)})
    # profit_p_daily
    profit_p = round(profit / 10000 - 1, 5)
    days = record.get('key').get('count_all')
    record.get('trade').update({'profit_p': round(np.log(1 + profit_p) / days * 252 * 100, 5)})
    if profit_last > 0:
        record.get('trade').update({'profit_p_daily': round(profit / profit_last * 100 - 100, 5)})
    # draw
    profit_max = record.get('trade').get('profit_max')
    record.get('trade').update({'draw': round(profit_max - profit, 5)})
    record.get('trade').update({'draw_p': round((1 - profit / profit_max) * 100, 5)})

    return None


# 根据新信息更新记录
def update_record(path_out, list_new_price, list_scale, pic):
    record = init(start='book' if path_out[:-8]=='test' else 'begin')

    # 开始更新
    report_of_record = pd.DataFrame()
    for i in np.arange(len(list_new_price)):
        # 价格读入
        new_price = list_new_price.iloc[i, :][0]
        new_price_trade = list_new_price.iloc[i, :][1]
        # 参数更新
        s1 = list_scale['s1'][i]
        s2 = list_scale['s2'][i]
        s3 = list_scale['s3'][i]
        # 匹配规则
        # match_rule(new_price, record, s1=4.6, s2=2.9, s3=1.6)
        match_rule(new_price, record, s1=s1, s2=s2, s3=s3)
        # 模拟交易
        sim_trade(new_price_trade, record, pic)
        # 计算交易
        calculate_trade(new_price, record)
        # 输出日志报告
        report_of_record = out_csv(new_price, report_of_record, record, i)

    # 打印图表
    if pic:
        out_log(record)
        report_of_record.to_csv(path_out, encoding='utf-8-sig')

    # 输出小结
    result = out_img(report_of_record, pic, path_out)

    return result


# 打印日志
def out_log(record):
    with open('update_record_log.txt', 'a', encoding='utf-8-sig') as f:
        for k, v in record.items():
            f.write('{} {}\n'.format(k, v))
        f.write('-----------\n')


# 数据报告
def out_csv(new_price, report_of_record, record, i):
    i += 1

    for column in ['ciup', 'ziup', 'up', 'down', 'zidown', 'cidown']:
        report_of_record.loc['0', column] = ''
    report_of_record.loc[i, record.get('last').get('column')] = record.get('last').get('new_price')

    report_of_record.loc[i, 'new_price'] = new_price
    report_of_record.loc[i, 'record_time'] = record.get('last').get('time')

    for column in record.get('key').keys():
        report_of_record.loc[i, column] = record.get('key').get(column)

    for column in record.get('sig').keys():
        report_of_record.loc[i, column] = str(record.get('sig').get(column))

    for column in record.get('trade').keys():
        report_of_record.loc[i, column] = record.get('trade').get(column)

    # report_of_record.to_csv(path_out, encoding='utf-8-sig')

    return report_of_record


# 打印图表
def out_img(report_of_record, pic, title):
    cols = ['new_price', 'buy_price_in', 'buy_price_out',
            'count_all', 'count_at', 'count_trade', 'count_succ', 'buy_in',
            'profit_p', 'draw_p', 'profit_p_daily', 'buy_profit_p',
            ]

    df = report_of_record.loc[:, cols]

    result = pd.DataFrame()
    result['all'] = [df['count_all'].max()]
    result['signal'] = [df['count_at'].max()]
    result['trade'] = [df['count_trade'].max()]
    result['win'] = [df['count_succ'].max()]
    result['profit'] = [round(df['profit_p'].iloc[-1], 2)]
    result['md'] = [round(df['draw_p'].max(), 2)]
    if df['draw_p'].max():
        result['pmd'] = [round(df['profit_p'].iloc[-1] / df['draw_p'].max(), 2)]
    else:
        result['pmd'] = None
    no_risk_r = np.e ** (0.03 / 365) - 1
    if df['profit_p_daily'].std():
        result['sr'] = [round((df['profit_p_daily'].mean() - no_risk_r) / df['profit_p_daily'].std() * np.sqrt(252), 2)]
    else:
        result['sr'] = None

    info = 'all:{}, signal:{}, trade:{}, win:{}, buy_in:{}, profit:{}%, md:{}%, P/MD:{}, SR:{}'.format(
        result['all'][0],
        result['signal'][0],
        result['trade'][0],
        result['win'][0],
        df['buy_in'].iloc[-1],
        result['profit'][0],
        result['md'][0],
        result['pmd'][0],
        result['sr'][0],
    )

    print(info)

    if pic:
        out_log({'info': info})
        plt.figure(figsize=(20, 10))
        df.loc[:, ['new_price', 'buy_price_in', 'buy_price_out']].plot(ax=plt.subplot(221), legend=True)
        df['buy_profit_p'].plot(ax=plt.subplot(222), legend=True)
        df['draw_p'] = -df['draw_p']
        df['draw_p'].plot(ax=plt.subplot(223), legend=True)
        df['profit_p'].plot(ax=plt.subplot(224), legend=True)

        plt.title(title)
        plt.savefig(str(time.time()) + str(title) + str(np.random.randint(100)) + '.png')
        plt.show()

    return result


# 存储中间
def out_mid(p1, p2, p3, mid_result, path):
    result = pd.concat(
        [pd.DataFrame([{'s1': p1, 's2': p2, 's3': p3}]), mid_result],
        axis=1,
        sort=False
    )
    result.to_csv(path, mode='a', encoding='utf-8-sig', header=False, index=False)


# 参数图表
def out_para(path, pp):
    df = pd.read_csv(path)
    df.reset_index(drop=True, inplace=True)

    # excel
    df.sort_values(by=pp, ascending=True, inplace=True)
    df['winr'] = round(df['win'] / df['trade'] * 100, 2)
    df = df[df['all'] == 1019]

    path_out = pd.ExcelWriter(path + '_out.xlsx')
    for col in df.columns[3:]:
        pd.pivot_table(df, values=col, index=pp[0], columns=pp[1], aggfunc=np.mean).to_excel(path_out, sheet_name=col)
    path_out.save()

    return None


# 3d画图
def out_img_3d(path, para_pool=None):
    df = pd.read_csv(path)
    df.reset_index(drop=True, inplace=True)
    df.columns = ['s1', 's2', 's3', 'all', 'signal', 'trade', 'win', 'profit', 'md', 'pmd', 'sr']
    # df = df[df['all'] == 1019]
    # np.where(df['profit'] > df['profit'].quantile(0.99), df['profit'], 0)

    # diy nots of 3d plotting
    df.sort_values(by=['s1', 's2', 's3'], ascending=True, inplace=True)

    if not para_pool:
        para_pool = {
            's1': np.arange(0, 5, 1),
            's2': np.arange(0, 5, 1),
            's3': np.arange(0, 5, 1),
        }

    x = para_pool.get('s1')
    y = para_pool.get('s2')
    z = para_pool.get('s3')

    X, Y, Z = np.meshgrid(y, x, z)  # yes, that's it: y, x, z crazy!!!
    print(df.shape, X.shape, Y.shape, Z.shape)

    for metric in ['trade', 'profit', 'md', 'pmd', 'sr']:
        c = df[metric] * 1000
        C = np.array(c)
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.set_xlabel(xlabel='s2')
        ax1.set_ylabel(ylabel='s1')
        ax1.set_zlabel(zlabel='s3')
        ax1.scatter(X, Y, Z, c=C, cmap=plt.get_cmap('coolwarm'))
        plt.legend(metric)
        plt.show()


# 查找参数
def find_para():
    path = 'diff-mean-123-0-5-0.5_endend_{}.csv'.format(str(time.time()))
    pn = ['s1', 's2', 's3']
    para_pool = {
        's1': np.arange(0, 5, 1),
        's2': np.arange(0, 5, 1),
        's3': np.arange(0, 5, 1),
    }

    pd.DataFrame(
        columns=['s1', 's2', 's3', 'all', 'signal', 'trade', 'win', 'profit', 'md', 'pmd', 'sr']
    ).to_csv(path, mode='w', encoding='utf-8-sig', header=True, index=False)

    for p1 in para_pool.get('s1'):
        for p2 in para_pool.get('s2'):
            for p3 in para_pool.get('s3'):
                para = {
                    's1': p1,
                    's2': p2,
                    's3': p3,
                }
                # out_log({'para': para})
                print(para)
                # out_mid(p1, p2, p3, run(path_in='601519-20122013.csv', how='endend', para=para), path)
                # out_mid(p1, p2, p3, run(path_in='601519-20122014.csv', how='endend', para=para), path)
                # out_mid(p1, p2, p3, run(path_in='601519-20122015.csv', how='endend', para=para), path)
                out_mid(p1, p2, p3, run(path='601519-20122016.csv', how='endend', para=para, pic=False), path)

    # 输出参数透视图或图表
    print(path)
    if len(pn) == 2:
        out_para(path, pp=pn[:2])
    elif len(pn) == 3:
        out_img_3d(path, para_pool)

    return None


# 从提取数据开始
def run(path='test.csv', how='endend', para=None, pic=False):
    path_out = [path[:-4] + '_out.csv' if '.csv' in path else path + '_out.csv'][0]
    # 读取数据
    df = pd.read_csv(path, header=None, skip_blank_lines=False)
    df.columns = ['end', 'beg', 'hig', 'low']

    if how == 'endend':
        df['new_price'] = df[how[:3]]
        df['new_price_trade'] = df[how[:-3]]

    list_new_price = df.loc[:, ['new_price', 'new_price_trade']]

    if para:
        df['s1'] = para.get('s1')
        df['s2'] = para.get('s2')
        df['s3'] = para.get('s3')

    else:
        df['s1'] = 4.6
        df['s2'] = 2.9
        df['s3'] = 1.6

    # 准备运行
    list_scale = df.loc[:, ['s1', 's2', 's3']]
    result = update_record(path_out, list_new_price, list_scale, pic)

    return result


# 开始运行
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.abspath(__file__)))


    # 寻找合适的模式下的合适的参数
    # find_para()
    # out_img_3d(path='')

    # 用寻找的合适的参数进行Li's模拟交易
    run(path='', how='endend', para=None, pic=True)
