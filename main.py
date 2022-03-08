import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pylab
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import kstest
import streamlit as st

figsize = (10, 4)


def main():
    data = load_data()
    st.title("Anabel Diss")

    show_data = st.sidebar.checkbox('Show Raw Data')
    if show_data:
        st.write(data)

    exclude_third_timestamp = st.sidebar.checkbox('Exclude third timestamp')
    if exclude_third_timestamp:
        data = data[data['Zeitpunkt_num'] != 3]

    exclude_third_gender = st.sidebar.checkbox('Exclude Patient Gender "mw"')
    if exclude_third_gender:
        data = data[data['sex_Pat'] != 'mw']

    y = st.sidebar.radio("Select Y Variable",
                        ('Ind_D_Kon_Mean', 'Ind_A_Kon_Mean'))
    x = st.sidebar.radio("Select X Category",
                        ('Klinik', 'Zeitpunkt_num', 'sex_Pat', 'sex_Arzt'))
    hue = st.sidebar.radio("Select Hue Category",
                        ('Klinik', 'Zeitpunkt_num', 'sex_Pat', 'sex_Arzt'))
    col = st.sidebar.radio("Select Column Category",
                                    ('Klinik', 'Zeitpunkt_num', 'sex_Pat',
                                     'sex_Arzt'))

    # dis_plot(data=data, hue=hue, x=y)
    hist_plot(data=data, hue=hue, x=y)
    # cat_plot(data=data, x=x, y=y, hue=hue, col=col)
    violin_plot(data=data, x=x, y=y, hue=hue)


@st.cache()
def load_data():
    csv_file = Path(__file__).parent / "anabel_master.csv"
    df = pd.read_csv(csv_file, sep=";")

    ind_cold = [col for col in df.columns if "Ind" in col]
    ind_d_kon_cols = [col for col in ind_cold if "D_Kon" in col]
    ind_a_kon_cols = [col for col in ind_cold if "A_Kon" in col]

    df['Ind_D_Kon_Mean'] = df[ind_d_kon_cols].mean(axis=1)
    df['Ind_A_Kon_Mean'] = df[ind_a_kon_cols].mean(axis=1)

    return df


def violin_plot(data, x, y, hue):
    fig = plt.figure(figsize=figsize)
    sns.violinplot(x=x, y=y, hue=hue, kind="swarm", data=data)
    st.pyplot(fig)


def dis_plot(data, hue, x):
    fig = plt.figure(figsize=figsize)
    g = sns.displot(kde=True, multiple='dodge', shrink=.8, hue=hue, x=x,
                    data=data)
    fig = g.ax
    # st.pyplot(fig)


def hist_plot(data, hue, x):
    fig = plt.figure(figsize=figsize)
    sns.histplot(kde=True, multiple='dodge', shrink=.8, hue=hue, x=x, data=data)
    st.pyplot(fig)


def cat_plot(data, x, y, hue, col, kind='violin', inner='quartile'):
    fig = plt.figure(figsize=figsize)
    sns.catplot(x=x, y=y, hue=hue, col=col, data=data, kind=kind, split=True,
                inner=inner, height=4, aspect=.7)
    st.pyplot(fig)


def calc_metrics(data):
    shapiro_test = False
    if shapiro_test:
        metric_data = data[data['Klinik'] == 'KinderKardio']['Ind_D_Kon_Mean']
        stat, p = shapiro(metric_data)
        print('stat=%.3f, p=%.3f\n' % (stat, p))
        if p > 0.05:
            print('Probably Gaussian')
        else:
            print('Probably not Gaussian')

    Kolmogorov_test = True
    if Kolmogorov_test:
        metric_data = data[data['Klinik'] == 'KinderKardio']['Ind_D_Kon_Mean']
        statistic, pvalue = kstest(metric_data, 'norm')
        print('statistic=%.3f, p=%.3f\n' % (statistic, pvalue))
        if pvalue > 0.05:
            print('Probably Gaussian')
        else:
            print('Probably not Gaussian')


if __name__ == '__main__':
    main()
