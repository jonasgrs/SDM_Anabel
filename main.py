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
# import plotly.express as px
import statsmodels.api as sm


figsize = (12, 4)
st.set_page_config(layout="wide")
sns.color_palette("Paired")
sns.set_palette('Paired')


# gray = '#7d7d7d'
# rc = {'figure.figsize':(8,4.5),
#           'axes.facecolor': gray,
#           'axes.edgecolor': gray,
#           'axes.labelcolor': 'white',
#           'figure.facecolor': gray,
#           'patch.edgecolor': gray,
#           'text.color': 'white',
#           'xtick.color': 'white',
#           'ytick.color': 'white',
#           'grid.color': 'grey',
#           'font.size' : 8,
#           'axes.labelsize': 12,
#           'xtick.labelsize': 8,
#           'ytick.labelsize': 12}
# plt.rcParams.update(rc)

def main():
    data = load_data()
    st.title("Anabel Diss")

    # --- Sidebar
    show_data = st.sidebar.checkbox('Show Raw Data')
    if show_data:
        st.write(data)

    exclude_third_timestamp = st.sidebar.checkbox('Exclude third timestamp')
    if exclude_third_timestamp:
        data = data[data['Zeitpunkt_num'] != 3]

    exclude_third_gender = st.sidebar.checkbox('Exclude Patient Gender "mw"')
    if exclude_third_gender:
        data = data[data['sex_Pat'] != 'mw']

    cats = ('Klinik', 'Zeitpunkt_num', 'sex_Pat', 'sex_Arzt', 'Rater_1',
            'Rater_2', 'Gespr_Art')
    vars = ('Ind_D_Kon_Mean', 'Ind_A_Kon_Mean', 'Gespr_Dauer in Sek.')
    st.sidebar.header('Combined Graphs')
    y = st.sidebar.radio("Select Variable", vars )
    x = st.sidebar.radio("Select X Category", cats)
    hue = st.sidebar.radio("Select Hue Category", cats)

    with st.expander("Show raw data"):
        st.write(data)

    cont1 = st.container()
    cont1.header('Statistical Metrics')

    col1l, col1m, col1r = st.columns(3)

    cont2 = st.container()
    # cont2.markdown("""---""")
    cont2.header('Combined Graphs')

    col2l, col2r = st.columns(2)

    cont3 = st.container()
    cont3.header('Individual Indicators')

    col3l, col3r = st.columns(2)


    with col1l:
        var = st.radio('Select variable for Metrics Calculation',
                       options=['Ind_D_Kon_Mean', 'Ind_A_Kon_Mean'])

        klinik = st.radio('Select Klinik for Metrics Calculation',
                          options=['KinderKardio', 'KiJuMed', 'Both'])

        p_kolmogorov = kolmogorov_test(data, klinik=klinik, var=var)
        p_shaprio = shapiro_test(data, klinik=klinik, var=var)

    with col1m:
        st.metric(label="Kolmogorov P-Value", value=p_kolmogorov,
                  delta=str(round(p_kolmogorov - 0.05, 5)) + " from 0.05")

    with col1r:
        st.metric(label="Shapiro P-Value", value=p_shaprio,
                  delta=str(round(p_shaprio - 0.05, 5)) + " from 0.05")

    with col2l:
        # cat_plot(data=data, x=x, y=y, hue=hue, col=col)
        violin_plot(data=data, x=x, y=y, hue=hue)

        qq_plot(data=data, var=y)

    with col2r:
        # dis_plot(data=data, hue=hue, x=y)
        hist_plot(data=data, hue=hue, x=y)

    with col3l:
        indicators = st.multiselect(
            'Select Indicators',
            [col for col in data.columns if "_Kon" in col and 'P_Kon' not in
             col])
    with col3r:
        var_cat = st.radio('Select variable category',
                       options=['D_Kon', 'A_Kon', 'Both'])

    plot_melted_histplot(data, indicators, var_cat)

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


def plot_melted_histplot(data, indicators=None, var_cat='_Kon'):
    if var_cat == 'Both':
        var_cat = '_Kon'
        figsize = (12, 8)
    else:
        figsize = (12, 4)
    if not indicators:
        indicators = [col for col in data.columns if var_cat in col]
        indicators = [col for col in indicators if "Mean" not in col]
        indicators = [col for col in indicators if "_P_Kon" not in col]

    data_to_melt = data[indicators]
    dfm = data_to_melt.melt(var_name='Indicators', value_name='Points')

    fig = plt.figure(figsize=figsize)
    sns.histplot(
        data=dfm,
        y="Indicators",
        binwidth=0.5,
        hue='Points',
        linewidth=.5,
        palette='flare',
        multiple="stack")
    st.pyplot(fig)


def qq_plot(data, var):
    fig = plt.figure()
    fig = sm.qqplot(data[var], line='45')
    st.pyplot(fig)

def violin_plot(data, x, y, hue):
    fig = plt.figure(
        # figsize=figsize
    )
    if len(data[hue].unique()) <= 2:
        split = True
    else:
        split = False

    sns.violinplot(x=x, y=y, hue=hue, kind="swarm", data=data, split=split,
                   scale='count', bw=.3, inner='quartile')

    st.pyplot(fig)


def dis_plot(data, hue, x):
    fig = plt.figure(
        # figsize=figsize
    )
    g = sns.displot(kde=True, multiple='dodge', shrink=.8, hue=hue, x=x,
                    data=data)
    fig = g.ax
    # st.pyplot(fig)


def hist_plot(data, hue, x):
    fig = plt.figure(
        # figsize=figsize
    )
    sns.histplot(kde=True, multiple='dodge', shrink=.8, hue=hue, x=x, data=data)

    # fig = px.histogram(data, x=x, y="Ind_A_Kon_Mean", color=hue,
    #                    marginal="box",  # or violin, rug
    #                    hover_data=data.columns)

    st.pyplot(fig)


def cat_plot(data, x, y, hue, col, kind='violin', inner='quartile'):
    fig = plt.figure(
        # figsize=figsize
    )
    sns.catplot(x=x, y=y, hue=hue, col=col, data=data, kind=kind, split=True,
                inner=inner, height=4, aspect=.7)
    st.pyplot(fig)


def shapiro_test(data, klinik, var):
    if len(klinik) == 1 and type(klinik) == list:
        metric_data = data[data['Klinik'] == klinik[0]][var]
    elif type(klinik) == str and klinik != 'Both':
        metric_data = data[data['Klinik'] == klinik][var]
    elif len(klinik) > 1 and type(klinik) == list or klinik == "Both":
        metric_data = data[var]
    else:
        raise Exception('Wrong input')

    stat, p = shapiro(metric_data)
    print('stat=%.3f, p=%.3f\n' % (stat, p))
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')

    return round(p, ndigits=5)


def kolmogorov_test(data, klinik, var):
    if len(klinik) == 1 and type(klinik) == list:
        metric_data = data[data['Klinik'] == klinik[0]][var]
    elif type(klinik) == str and klinik != 'Both':
        metric_data = data[data['Klinik'] == klinik][var]
    elif len(klinik) > 1 and type(klinik) == list or klinik == "Both":
        metric_data = data[var]
    else:
        raise Exception('Wrong input')

    statistic, p = kstest(metric_data, 'norm')
    print('statistic=%.3f, p=%.3f\n' % (statistic, p))
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')

    return round(p, ndigits=5)


if __name__ == '__main__':
    main()
