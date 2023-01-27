import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats

# Title
st.title("Bacteria Identification using Metabolic Profile")
st.subheader("- by Pahvindran Raj")

# gif from local file
file_ = open("assets/bacteria1.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="bacteria1 gif">',
    unsafe_allow_html=True,
)
st.header('')
# bacterial identification of  Klebsiella pneumoniae and Burkholderia pseudomallei
st.header('Identifcation of bacterial species using :blue[discrete inputs]:')

tab_1, tab_2 = st.tabs(['Choose metabolites directly', 'List down metabolites'])
with tab_1:
    discrete_input = np.zeros((1, 27), dtype=int)
    left_column, right_column = st.columns(2)
    if left_column.checkbox('BCAA'): discrete_input[:, 1] = 1
    if left_column.checkbox('Ethanol'): discrete_input[:, 1] = 1
    if left_column.checkbox('Threonine'): discrete_input[:, 2] = 1
    if left_column.checkbox('Alanine'): discrete_input[:, 3] = 1
    if left_column.checkbox('Lysine'): discrete_input[:, 4] = 1
    if left_column.checkbox('Acetate'): discrete_input[:, 5] = 1
    if left_column.checkbox('Methionine'): discrete_input[:, 6] = 1
    if left_column.checkbox('Pyruvate'): discrete_input[:, 7] = 1
    if left_column.checkbox('Succinate'): discrete_input[:, 8] = 1
    if left_column.checkbox('TMAO'): discrete_input[:, 9] = 1
    if left_column.checkbox('Gylcerophosphocholine'): discrete_input[:, 10] = 1
    if left_column.checkbox('Phenylacetate'): discrete_input[:, 11] = 1
    if left_column.checkbox('Glycine'): discrete_input[:, 12] = 1
    if left_column.checkbox('Valine'): discrete_input[:, 13] = 1
    if right_column.checkbox('Uracil'): discrete_input[:, 14] = 1
    if right_column.checkbox('Fumarate'): discrete_input[:, 15] = 1
    if right_column.checkbox('Tyrosine'): discrete_input[:, 16] = 1
    if right_column.checkbox('Xanthine'): discrete_input[:, 17] = 1
    if right_column.checkbox('Hypoxanthine'): discrete_input[:, 18] = 1
    if right_column.checkbox('Formate'): discrete_input[:, 19] = 1
    if right_column.checkbox('Butanoate'): discrete_input[:, 20] = 1
    if right_column.checkbox('3-hydroxybutanoate'): discrete_input[:, 21] = 1
    if right_column.checkbox('Putrescine'): discrete_input[:, 22] = 1
    if right_column.checkbox('Acetone'): discrete_input[:, 23] = 1
    if right_column.checkbox('Creatine'): discrete_input[:, 24] = 1
    if right_column.checkbox('Methanol'): discrete_input[:, 25] = 1
    if right_column.checkbox('Phenylalanine'): discrete_input[:, 26] = 1

    # load discrete model
    st.write('')
    if st.button('classify'):
        filename = 'models/svm_model_discrete.sav'
        model = pickle.load(open(filename, 'rb'))
        pred = model.predict(discrete_input)
        output = ''
        if pred[0] == 0:
            output = 'Klebsiella pneumoniae'
        if pred[0] == 1:
            output = 'Burkholderia pseudomallei'

        output_text = st.text_input('', value=output)
with tab_2:
    st.write('List down metabolites separated by ","')
    text_input = st.text_area('Example: ', value='Acetone,Formate,Butanoate,Creatine')

    discrete_input = np.zeros((1, 27), dtype=int)
    metabolites_input_arr = text_input.split(",")
    error_found = False
    for metabolite in metabolites_input_arr:
        metabolite = metabolite.strip().lower()
        if metabolite == "bcaa": discrete_input[:, 0] = 1
        elif metabolite == "ethanol": discrete_input[:, 1] = 1
        elif metabolite == "threonine": discrete_input[:, 2] = 1
        elif metabolite == "alanine": discrete_input[:, 3] = 1
        elif metabolite == "lysine": discrete_input[:, 4] = 1
        elif metabolite == "acetate": discrete_input[:, 5] = 1
        elif metabolite == "methionine": discrete_input[:, 6] = 1
        elif metabolite == "pyruvate": discrete_input[:, 7] = 1
        elif metabolite == "succinate": discrete_input[:, 8] = 1
        elif metabolite == "tmao": discrete_input[:, 9] = 1
        elif metabolite == "gylcerophosphocholine": discrete_input[:, 10] = 1
        elif metabolite == "phenylacetate": discrete_input[:, 11] = 1
        elif metabolite == "glycine": discrete_input[:, 12] = 1
        elif metabolite == "valine": discrete_input[:, 13] = 1
        elif metabolite == "uracil": discrete_input[:, 14] = 1
        elif metabolite == "fumarate": discrete_input[:, 15] = 1
        elif metabolite == "tyrosine": discrete_input[:, 16] = 1
        elif metabolite == "xanthine": discrete_input[:, 17] = 1
        elif metabolite == "hypoxanthine": discrete_input[:, 18] = 1
        elif metabolite == "formate": discrete_input[:, 19] = 1
        elif metabolite == "butanoate": discrete_input[:, 20] = 1
        elif metabolite == "3-hydroxybutanoate": discrete_input[:, 21] = 1
        elif metabolite == "putrescine": discrete_input[:, 22] = 1
        elif metabolite == "acetone": discrete_input[:, 23] = 1
        elif metabolite == "creatine": discrete_input[:, 24] = 1
        elif metabolite == "methanol": discrete_input[:, 25] = 1
        elif metabolite == "phenylalanine": discrete_input[:, 26] = 1
        else:
            if not error_found:
                st.error('The system could not detect the following metabolite(s):', icon="🚨")
            error_found = True
            st.error(metabolite)

    # load discrete model
    st.write('')
    if st.button('classify '):
        if error_found:
            st.warning('System will ignore the unidentified metabolites', icon="⚠️")
        filename = 'models/svm_model_discrete.sav'
        model = pickle.load(open(filename, 'rb'))
        pred = model.predict(discrete_input)
        output = ''
        if pred[0] == 0:
            output = 'Klebsiella pneumoniae'
        if pred[0] == 1:
            output = 'Burkholderia pseudomallei'

        output_text = st.text_input('', value=output)

st.header('')
# bacterial identification of using NMR spectra
st.header('Identifcation of bacteria using :blue[NMR spectra]:')
st.write('Upload NMR spectra dataset. You may leave the label column empty if you only want to do the prediction.')
dataset_display = pd.read_csv('datasets/dataset_display.csv')
st.dataframe(dataset_display)
uploaded_file = st.file_uploader("Choose a file")
dataset = None
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)

with st.expander('You may download and use the template of the NMR spectra dataset'):
    st.write('')
    st.write('The following template is a 1D^1H NMR spectroscopy. The dataset does not necessarily need to match the'
             ' same chemical shift interval (ppm) as the program will output the interval as the label. However, if you wish to'
             ' use the pretrained model the chemical shifts need to match.')
    csv = open('datasets/dataset_template.csv')
    st.download_button(
        label="Download template as CSV",
        data=csv,
        file_name='dataset_template.csv',
        mime='text/csv',
        )
    csv.close()

# split dataset inputs into X and y
def inputs_normal(dataset):
    X = dataset.loc[:, dataset.columns != 'Label'].values
    y = dataset.loc[:, ['Label']].values.flatten()
    return X, y

# split dataset inputs into X and y (increasing signals)
def inputs_incsignals(dataset):
    X = dataset.loc[:, dataset.columns != 'Label'].values
    X[X < 0] = 0
    y = dataset.loc[:, ['Label']].values.flatten()
    return X, y

# Principal Component Analysis
st.write('')
st.subheader('Principal Component Analysis (PCA)')
tab1, tab2 = st.tabs(['Normal', 'Using increasing signals'])
with tab1:
    if dataset is not None:
        X, y = inputs_normal(dataset)
        pca = PCA(n_components=2)
        principal_comps = pca.fit_transform(X)
        pc_df = pd.DataFrame(data=principal_comps, columns=['PC1', 'PC2'])
        pc_df = pd.concat([pc_df, dataset.loc[:, ['Label']]], axis=1)

        # plot the graph
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('2 component PCA')
        targets = pd.DataFrame(y).drop_duplicates().loc[:,0].values
        for target in targets:
            indicesToKeep = pc_df['Label'] == target
            ax.scatter(pc_df.loc[indicesToKeep, 'PC1'], pc_df.loc[indicesToKeep, 'PC2'])
        ax.legend(targets)
        ax.grid()
        # ax.patch.set_facecolor('black')
        # fig.patch.set_facecolor('grey')
        st.pyplot(fig)
        st.write('PCA explained variance ratio, PC1:PC2: {}'.format(pca.explained_variance_ratio_))
    else:
        st.warning('No dataset found', icon="⚠️")
with tab2:
    if dataset is not None:
        X, y = inputs_incsignals(dataset)
        pca = PCA(n_components=2)
        principal_comps = pca.fit_transform(X)
        pc_df = pd.DataFrame(data=principal_comps, columns=['PC1', 'PC2'])
        pc_df = pd.concat([pc_df, dataset.loc[:, ['Label']]], axis=1)

        # plot the graph
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('2 component PCA')
        targets = pd.DataFrame(y).drop_duplicates().loc[:,0].values
        for target in targets:
            indicesToKeep = pc_df['Label'] == target
            ax.scatter(pc_df.loc[indicesToKeep, 'PC1'], pc_df.loc[indicesToKeep, 'PC2'])
        ax.legend(targets)
        ax.grid()
        st.pyplot(fig)
        st.write('PCA explained variance ratio, PC1:PC2: {}'.format(pca.explained_variance_ratio_))
    else:
        st.warning('No dataset found', icon="⚠️")

# Bacteria Prediction
st.write('')
st.subheader('Bacteria Prediction')
st.write('You can train your own model or use the existing pretrained model for the prediction.')
with st.expander('details about the pretrained model'):
    st.write('The pretrained model was trained on 1D^1H NMR spectroscopy (same as the template) and could predict the following bacteria:')
    st.write('1) Bacillus')
    st.write('2) Candida')
    st.write('3) EColi-K12')
    st.write('4) EColi-O157H7')
    st.write('5) EColi-K12')
    st.write('6) Listeria')
    st.write('7) Pseudomonas')
    st.write('8) Salmonella')
    st.write('9) Shingella')
    st.write('10) Staphylococcus')
    st.write('11) Yersinia')
tab1_1, tab2_1 = st.tabs(['Model Training', 'Prediction'])
rf = None
with tab1_1:
    left_column, right_column = st.columns(2)
    if dataset is not None:
        X, y = inputs_normal(dataset)
        left_column.checkbox('normal (default)')
        if right_column.checkbox('use increasing signals'):
            X, y = inputs_incsignals(dataset)
        st.write('')
        if st.button('Train Model'):
            rf = RandomForestClassifier(n_estimators=100, random_state=123)
            rf.fit(X, y)
            st.success('Model has been trained!', icon="✅")

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

with tab2_1:
    if st.checkbox('use pretrained model'):
        filename = 'models/rf_model_nmr.sav'
        rf = pickle.load(open(filename, 'rb'))
    if dataset is not None:
        if rf is None:
            st.warning('Please train a model or choose the pretrained model.', icon="⚠️")
        else:
            pred = rf.predict(X)
            output_csv = convert_df(pd.DataFrame(pred))
            st.download_button(
                label="Download output as CSV",
                data=output_csv,
                file_name='predictions.csv',
                mime='text/csv',
            )
            with st.expander('show predictions as a dataframe'):
                st.dataframe(pd.DataFrame(pred))

# Bacteria Fold Analysis
st.write('')
st.subheader('Bacteria Signal/Fold Analysis')
if dataset is not None:
    _, y = inputs_normal(dataset)
    # dataset.loc[:, ['Label']].values.flatten()
    bacterium = pd.DataFrame(y).drop_duplicates().loc[:,0]
    bacteria_selected = st.selectbox('Choose a bacteria from the dataset', bacterium)

    bacteria = dataset.loc[(dataset['Label'] == bacteria_selected), dataset.columns != 'Label'].values

    st.write('Metabolites that has significant change (increase or decrease) during the incubation is idenfied using one'
             ' sample t-test.')

    p_value_cutoff = 0.05

    bacteria_inc = bacteria.copy()
    bacteria_dec = bacteria.copy()
    bacteria_inc[bacteria_inc < 0] = 0
    bacteria_dec[bacteria_dec > 0] = 0

    # perform one sample t-test
    _, p_value_inc = stats.ttest_1samp(a=bacteria_inc, popmean=0)
    _, p_value_dec = stats.ttest_1samp(a=bacteria_dec, popmean=0)

    p_val_inc_index = []
    p_val_dec_index = []
    for index in range(p_value_inc.size):
        if p_value_inc[index] <= p_value_cutoff:
            p_val_inc_index.append(index)
        if p_value_dec[index] <= p_value_cutoff:
            p_val_dec_index.append(index)

    p_val_inc_index = np.array(p_val_inc_index)
    p_val_dec_index = np.array(p_val_dec_index)

    # bacteria_inc_signal = pd.DataFrame(bacteria_inc)
    # bacteria_dec_signal = pd.DataFrame(bacteria_dec)
    # for column_index in range(p_val_inc_index.size):
    #     if p_val_inc_index[column_index] != column_index:
    #         bacteria_inc_signal = bacteria_inc_signal.drop([bacteria_inc_signal.columns[column_index]], axis=1)
    # for column_index in range(p_val_dec_index.size):
    #     if p_val_dec_index[column_index] != column_index:
    #         bacteria_dec_signal = bacteria_dec_signal.drop([bacteria_dec_signal.columns[column_index]], axis=1)

    tab1_2, tab2_2 = st.tabs(['Graph', 'Output'])
    with tab1_2:
        # plotting the graph
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Chemical Shift (ppm)')
        ax.set_ylabel('Metabolite Frequency')
        ax.set_title(bacteria_selected)

        x = np.linspace(0.02, 9.5, bacteria.shape[1])
        y_zeros = np.linspace(0, 0, bacteria.shape[1])
        ax.scatter(x, bacteria_inc[0], c='r', marker='+')
        ax.scatter(x, bacteria_dec[0], c='b', marker='x')
        ax.scatter(x, y_zeros, c='black', marker=',')
        targets = ['Increasing signal', 'Decreasing signal', 'Mean']
        ax.legend(targets)
        ax.grid()
        st.pyplot(fig)
    with tab2_2:
        st.write('')
        bacteria_inc_signal = pd.DataFrame(dataset.columns[1:]).transpose()
        bacteria_dec_signal = pd.DataFrame(dataset.columns[1:]).transpose()
        columns_to_drop = []
        for column_index in range(p_val_inc_index.size):
            if p_val_inc_index[column_index] != column_index:
                columns_to_drop.append(bacteria_inc_signal.columns[column_index])
        bacteria_inc_signal = bacteria_inc_signal.drop(columns_to_drop, axis=1)

        columns_to_drop2 = []
        for column_index2 in range(p_val_dec_index.size):
            if p_val_dec_index[column_index2] != column_index2:
                columns_to_drop2.append(bacteria_inc_signal.columns[column_index])
        bacteria_dec_signal = bacteria_dec_signal.drop(columns_to_drop2, axis=1)

        # Download/display output
        output_incsignal_csv = convert_df(bacteria_inc_signal)
        output_decsignal_csv = convert_df(bacteria_dec_signal)
        left_column, right_column = st.columns(2)
        with left_column:
            st.download_button(
                label="Download increasing signal columns as CSV",
                data=output_incsignal_csv,
                file_name='increasing_signal_columns.csv',
                mime='text/csv',
            )
        with right_column:
            st.download_button(
                label="Download decreasing signal columns as CSV",
                data=output_decsignal_csv,
                file_name='decreasing_signal_columns.csv',
                mime='text/csv',
            )
        with st.expander('show increasing and decreasing signals as dataframe'):
            st.write('Increasing Signal Columns')
            st.dataframe(bacteria_inc_signal)
            st.write('Decreasing Signal Columns')
            st.dataframe(bacteria_dec_signal)

else:
    st.warning('No dataset found', icon="⚠️")
