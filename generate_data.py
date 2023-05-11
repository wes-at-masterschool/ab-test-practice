# -*- coding: utf-8 -*-
"""
Created on Thu May 11 06:18:22 2023

@author: wes_c
"""

import numpy.random as random
import pandas as pd
import statsmodels.stats.proportion as proportion

from scipy.stats import ttest_ind

def biased_coin(p):
    if random.uniform(0, 1) < p:
        return True
    else:
        return False

def generate_conv(p):
    return biased_coin(p)

def get_value(lst):
    return random.choice(lst)

def get_cts_value(bnds):
    return random.uniform(*bnds)

def generate_row(a_conv=0.1, b_conv=0.11, a_spend=[2,4], b_spend=[2.01, 4.08],
                 devices=['iPhone', 'Android', 'PC', 'Mac', 'Linux'],
                 gender=['M', 'F', 'Other'], region=['NA', 'SA', 'EU', 'AS', 'OC', 'AF']):
    group = get_value(['A', 'B'])
    row = [group, get_value(devices), get_value(gender), get_value(region)]
    if group == 'A':
        converted = generate_conv(a_conv)
        row.append(converted)
        if converted:
            row.append(get_cts_value(a_spend))
        else:
            row.append(0)
    else:
        converted = generate_conv(b_conv)
        row.append(converted)
        if converted:
            row.append(get_cts_value(b_spend))
        else:
            row.append(0)
    return row

def make_data(n=10000, **kwargs):
    rows = []
    for _ in range(n):
        rows.append(generate_row(**kwargs))
    df = pd.DataFrame(rows, columns=['group', 'device', 'gender', 'region',
                                     'converted', 'spend'])
    return df

def chi_square(df):
    df['converted'] = df['converted'].apply(int)
    gdf = df.groupby('group').sum()['converted']
    sdf = df.groupby('group').size()
    n_obs = [gdf[0], gdf[1]]
    sizes = [sdf[0], sdf[1]]
    chi2stat, pval1, table1 = proportion.proportions_chisquare(n_obs, sizes)
    zstat, pval2 = proportion.proportions_ztest(n_obs, sizes)
    return chi2stat, pval1, zstat, pval2

def ttest(df):
    a = df[df['group'] == 'A']['spend']
    b = df[df['group'] == 'B']['spend']
    tstat, pval = ttest_ind(a, b, equal_var=False)
    return tstat, pval

if __name__ == '__main__':
    df = make_data()
    df.to_csv('ab_test1.csv', index=False)
    df = make_data(n=100000)
    df.to_csv('ab_test2.csv', index=False)
    df = make_data(n=1000000)
    df.to_csv('ab_test3.csv', index=False)
