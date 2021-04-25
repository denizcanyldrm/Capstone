import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, plot_roc_curve
import pickle
import os


def get_unknown_values_df(attributes_df):
    """ Gets rows of attributes_df with only contains unknown values

    Args:
        attributes_df (dataframe): a dataframe

    Returns:
        dataframe: a new dataframe with only unknown values
    """

    unknown_attr = attributes_df['Meaning'].where(attributes_df['Meaning'].str.contains('unknown')).value_counts().index
    attributes_unknown_values = attributes_df[attributes_df['Meaning'].isin(unknown_attr)]
    attributes_unknown_values.drop(['Description', 'Meaning'], axis=1, inplace=True)

    return attributes_unknown_values


def preprocess_unknown_values(df, attr_unknown_values):
    """ Converts dataframe's (df) unknown values into nan values

    Args:
        df (dataframe): a dataframe
        attr_unknown_values (dataframe): attribute dataframe with only unknown values

    Returns:
        None
    """

    i = 0
    for col in df.columns:
        # first column is not an attribute
        if i == 0:
            i += 1
            continue

        values_df = attr_unknown_values[attr_unknown_values['Attribute'] == col]

        # column that contain no unknown values
        if values_df.shape[0] == 0:
            continue

        unknown_values = str(values_df.iloc[0]['Value']).split(',')

        unknown_list = []
        for val in unknown_values:
            if val == 'X' or val == 'XX':
                unknown_list.append(str(val))
            else:
                unknown_list.append(int(val))

        df[col].replace({unk: np.nan for unk in unknown_list}, inplace=True)


def apply_column_specific_operations(df, ost_west_col, date_col, object_cols):
    """ Applies column specific operations of df dataframe with given columns as parameters

    Args:
        df (dataframe): a dataframe
        ost_west_col (string): 'OST_WEST_KZ' column of dataframe
        date_col (string): 'EINGEFUEGT_AM' column of dataframe
        object_cols (list): 'object' columns of dataframe

    Returns:
        None
    """

    df[ost_west_col].replace({'W': 1, 'O': 0}, inplace=True)
    df[ost_west_col] = df[ost_west_col].astype(float)

    df[date_col] = pd.to_datetime(df[date_col]).dt.year

    for col in object_cols:
        df[col] = df[col].astype(str)
        df[col].replace({'X': np.nan, 'XX': np.nan}, inplace=True)


def get_missing_attr_info(df):
    """ Calculates percentage of nan values by attributes

    Args:
        df (dataframe): a dataframe

    Returns:
        dataframe: a dataframe holds attribute and their nan percentages
    """

    attributes_by_missing_values = (df.isnull().sum() / df.shape[0]).sort_values(ascending=False).reset_index()
    attributes_by_missing_values.columns = ['Attribute', 'Null Percentage']

    return attributes_by_missing_values


def plot_missing_values(attr_by_missing_info, col_num, fig_size=(20, 5)):
    """ Plots attributes and percentages of dataframe

    Args:
        attr_by_missing_info (dataframe): a dataframe holds attribute and their nan percentages
        col_num (int): column number
        fig_size (tuple): figure size

    Returns:
        None
    """

    fig = plt.figure(figsize=fig_size)
    plt.bar(range(col_num), attr_by_missing_info['Null Percentage'][:col_num], align='center', color='purple')
    plt.xticks(range(col_num), attr_by_missing_info['Attribute'][:col_num], rotation='vertical')
    plt.title('Top {} Columns According To The Percentage Of Nan Values'.format(col_num))
    plt.xlabel('Attribute Name')
    plt.ylabel('Percentage')
    plt.show()


def drop_unnecessary_cols(df, attr_by_missing_info, is_train_data=True, drop_percentage=0.5):
    """ Finds and drops columns that have more than drop_percentage nan values

    Args:
        df (dataframe): a dataframe
        attr_by_missing_info (dataframe): dataframe holds attribute and their nan percentages
        is_train_data (bool): flag shows that is train data or not
        drop_percentage (float): percentage

    Returns:
        None
    """

    attr_dropped = list(attr_by_missing_info[attr_by_missing_info['Null Percentage'] > drop_percentage]['Attribute'])
    if is_train_data:
        attr_dropped.append('LNR')

    df.drop(attr_dropped, axis=1, inplace=True)


def drop_unnecessary_rows(df, drop_percentage=0.5):
    """ Finds and drops rows that have more than drop_percentage nan values

    Args:
        df (dataframe): a dataframe
        drop_percentage (float): percentage

    Returns:
        dataframe: a dataframe that holds index and missing percentages
    """

    missing_rows = ((df.isnull().sum(axis=1)) / df.shape[1]).sort_values(ascending=False).reset_index()
    missing_rows.columns = ['Index', 'Percentage']

    rows_dropped = list(missing_rows[missing_rows['Percentage'] > drop_percentage]['Index'])
    print('Number of rows that will be dropped: {}'.format(len(rows_dropped)))

    df.drop(rows_dropped, inplace=True)
    return missing_rows


def plot_hist_nan_rows(df, missing_rows, step=10, fig_size=(20, 5)):
    """ Plots the number of nan values vs the number of rows

    Args:
        df (dataframe): a dataframe
        missing_rows (dataframe): a dataframe that holds index and missing percentages
        step (int): step
        fig_size (tuple): figure size

    Returns:
        None
    """

    ranges = np.arange(min(missing_rows['Percentage'] * df.shape[1]), max(missing_rows['Percentage'] * df.shape[1]) + step, step)

    (missing_rows['Percentage'] * df.shape[1]).plot.hist(bins=ranges, alpha=1, color='purple', figsize=fig_size)
    plt.xticks(ranges)
    plt.title('Distribution of nan values among all rows')
    plt.xlabel('The number of nan values')
    plt.ylabel('The number of rows')
    plt.show()


def fit_transform_pipeline(df, object_col_list):
    """ Creates a pipeline and applies that pipeline to given dataframe

    Args:
        df (dataframe): a dataframe
        object_col_list (list): a list of object columns of dataframe

    Returns:
        dataframe: a new dataframe that is imputed and scaled
    """

    df[object_col_list] = df[object_col_list].apply(LabelEncoder().fit_transform)
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    df_data = pipe.fit_transform(df)
    df_transformed = pd.DataFrame(df_data, columns=list(df.columns))
    return df_transformed


def save_dataframe(df, dir_name, file_name):
    """ Saves dataframe into a file

    Args:
        df (dataframe): a dataframe
        dir_name (str): name of the directory
        file_name (str): name of the file

    Returns:
        None
    """

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    df.to_pickle(dir_name + '/' + file_name)


def load_dataframe(file_name):
    """ Loads dataframe from a file

    Args:
        file_name (str): name of the file

    Returns:
        dataframe: a dataframe is loaded from a file
    """

    df = pd.read_pickle(file_name)
    return df


def plot_array(data_x, data_y, title, x_label, y_label, fig_size=(20, 5)):
    """ Plots given data arrays

    Args:
        data_x (array): a numpy array for x axis
        data_y (array): a numpy array for y axis
        title (str): plot title
        x_label (str): x axis label
        y_label (str): y axis label
        fig_size (tuple): figure size

    Returns:
        None
    """

    fig = plt.figure(figsize=fig_size)
    plt.plot(data_x, data_y, color='purple')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def apply_pca(df, df_name):
    """ Applies PCA to a given dataframe

    Args:
        df (dataframe): a dataframe
        df_name (string): name of dataframe

    Returns:
        model: a PCA model
        dataframe: a new dataframe that is formed by fitting PCA model
    """

    pca = PCA(n_components=0.95, svd_solver='auto')
    df_pca = pca.fit_transform(df)
    df_pca = pd.DataFrame(df_pca)

    print('explained_variance_ratio_: {}  df.shape[1]: {}'.format(len(pca.explained_variance_ratio_), df.shape[1]))
    var_ratio_cum_sum = np.cumsum(pca.explained_variance_ratio_)

    title = 'PCA with n_components = 0.95 ({})'.format(df_name)
    x_label = 'Principal Components'
    y_label = 'Cumulative Explained Variance Ratio'
    plot_array(np.arange(len(var_ratio_cum_sum)), var_ratio_cum_sum, title, x_label, y_label)

    return pca, df_pca


def print_principle_component(variances, df_cols):
    """ Prints principal components by variances

    Args:
        variances (array): a numpy array
        df_cols (list): a list of column names

    Returns:
        None
    """

    df_feature_by_vars = pd.DataFrame(variances, index=df_cols, columns=['Variance'])
    df_feature_by_vars.sort_values(by='Variance', ascending=False, inplace=True)

    print(df_feature_by_vars.head())


def find_optimal_k(df_pca):
    """ Finds optimal k value

    Args:
        df_pca (dataframe): a dataframe

    Returns:
        integer: optimal k value
    """

    cluster_list = np.arange(1, 21)
    scores = []

    for k in cluster_list:
        k_means = KMeans(n_clusters=k)
        model = k_means.fit(df_pca)
        scores.append(k_means.inertia_)

    title = 'K-Means sum of square distances'
    x_label = '# of clusters'
    y_label = 'sum of square distances'
    plot_array(cluster_list, scores, title, x_label, y_label)

    optimum_k = list(scores > np.average(scores)).index(False) + 1

    return optimum_k


def preprocess_data(df, df_name, is_train_data, object_col_list, attributes_unknown_values):
    """ Applies preprocessing operations to given dataframe

    Args:
        df (dataframe): a dataframe
        df_name (str): name of dataframe
        is_train_data (bool): flag shows that data is train (true) or test (false)
        object_col_list (list): a list of object columns
        attributes_unknown_values (dataframe): a dataframe with  attributes and their unknown values

    Returns:
        None
    """

    # Apply specific operations to 'OST_WEST_KZ' and 'EINGEFUEGT_AM' columns
    apply_column_specific_operations(df, 'OST_WEST_KZ', 'EINGEFUEGT_AM', object_col_list)

    # Convert dataframe's unknown values into nan values
    preprocess_unknown_values(df, attributes_unknown_values)
    print(df.dtypes.value_counts())

    # Plot top 50 Columns According To The Percentage Of Nan Values
    attributes_by_missing_info_df = get_missing_attr_info(df)
    plot_missing_values(attributes_by_missing_info_df, 50)

    # Drop columns with having nan values more than 50% and 'LNR' column
    prev_df_shape = df.shape
    drop_unnecessary_cols(df, attributes_by_missing_info_df, is_train_data)
    print('{} shape changes from: {} to: {}'.format(df_name, prev_df_shape, df.shape))

    # Drop rows with having nan values more than 50%
    if is_train_data:
        prev_df_shape = df.shape
        missing_rows_df = drop_unnecessary_rows(df)
        print('{} shape changes from: {} to: {}'.format(df_name, prev_df_shape, df.shape))

        # Plot distribution of nan values among all rows
        plot_hist_nan_rows(df, missing_rows_df)

        df_x = df.drop(['RESPONSE'], axis=1)
        df_y = df['RESPONSE']
    else:
        df_x = df.drop(['LNR'], axis=1)
        df_y = None

    # Create a pipeline that applies LabelEncoder, SimpleImputer and StandardScaler to dataframe
    df_data_x = fit_transform_pipeline(df_x, object_col_list)

    if not is_train_data:
        df_data_x['LNR'] = df['LNR']

    # Save transformed data into a file
    save_dataframe(df_data_x, 'outputs', '{}_data_X.p'.format(df_name))
    if is_train_data:
        save_dataframe(df_y, 'outputs', '{}_data_y.p'.format(df_name))

    print(df_data_x[object_col_list].head())


def find_best_classifier(classifiers_dict, train_x, train_y, fig_size=(20, 10)):
    """ Applies preprocessing operations to given dataframe

    Args:
        classifiers_dict (dict): a dictionary of classifiers
        train_x (dataframe): preprocessed mailout train X data
        train_y (dataframe): mailout train y data
        fig_size (tuple): figure size

    Returns:
        str: name of the classifier that has the best performance
        float: best test performance score
    """

    clf_perf_dict = {'classifier': [],
                     'train_roc_auc_score': [],
                     'test_roc_auc_score': []
                     }

    # Split data into the train and test sets
    x_sub_train, x_sub_test, y_sub_train, y_sub_test = train_test_split(train_x, train_y, test_size=0.2, random_state=45)

    print('train_X.shape: {} train_y.shape: {}'.format(train_x.shape, train_y.shape))
    print('x_sub_train.shape: {} y_sub_train.shape: {}'.format(x_sub_train.shape, y_sub_train.shape))
    print('x_sub_test.shape: {} y_sub_test.shape: {}'.format(x_sub_test.shape, y_sub_test.shape))

    best_classifier_name = None
    best_classifier_score = 0

    # Iterate through classifiers and calculate their roc_auc_scores
    for clf in classifiers_dict.keys():
        classifiers_dict[clf].fit(x_sub_train, y_sub_train)

        train_score = roc_auc_score(y_sub_train, classifiers_dict[clf].predict_proba(x_sub_train)[:, 1])
        test_score = roc_auc_score(y_sub_test, classifiers_dict[clf].predict_proba(x_sub_test)[:, 1])

        clf_perf_dict['classifier'].append(classifiers_dict[clf])
        clf_perf_dict['train_roc_auc_score'].append(train_score)
        clf_perf_dict['test_roc_auc_score'].append(test_score)

        if test_score > best_classifier_score:
            best_classifier_score = test_score
            best_classifier_name = clf

    performances = pd.DataFrame.from_dict(clf_perf_dict, orient='index').T
    performances['classifier'] = list(classifiers_dict.keys())
    print('Performances:')
    print(performances)

    # Plot roc curve of classifiers
    fig, ax = plt.subplots(figsize=fig_size)
    for clf in classifiers_dict.keys():
        plot_name = clf + ' (train)'
        roc_curve_clf_train = plot_roc_curve(classifiers_dict[clf], x_sub_train, y_sub_train, ax=ax, alpha=0.8, name=plot_name)
        plot_name = clf + ' (test)'
        roc_curve_clf_test = plot_roc_curve(classifiers_dict[clf], x_sub_test, y_sub_test, ax=ax, alpha=0.8, name=plot_name)
    plt.show()

    return best_classifier_name, best_classifier_score


def perform_grid_search(train_x, train_y):
    """ Performs grid search on train data and saves the best model with parameters

    Args:
        train_x (dataframe): preprocessed mailout train X data
        train_y (dataframe): mailout train y data

    Returns:
        None
    """

    parameters_gbc = {
        'learning_rate': [0.1, 0.2, 0.3],
        'n_estimators': [50, 100],
        'loss': ['deviance', 'exponential']
    }

    gbc = GradientBoostingClassifier(random_state=45)
    cv_gbc = GridSearchCV(gbc, param_grid=parameters_gbc)

    cv_gbc.fit(train_x, train_y)
    print(cv_gbc.best_params_)

    best_gbc = cv_gbc.best_estimator_
    save_data(best_gbc, 'model', 'best_gbc.p')


def save_data(data, dir_name, file_name):
    """ Saves data into a file

    Args:
        data (model_list_dataframe): data
        dir_name (str): name of the directory
        file_name (str): name of the file

    Returns:
        None
    """

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(dir_name + '/' + file_name, 'wb') as f:
        pickle.dump(data, f)


def load_data(file_name):
    """ Loads data from a given file

    Args:
        file_name (str): name of the file

    Returns:
        (model_list_dataframe): data
    """

    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    return data


def plot_feature_importances(model, attributes, best_attr_num=10, fig_size=(20, 5)):
    """ Plots Top 10 attributes with importance selected by model

    Args:
        model (model): best GradientBoostingClassifier model
        attributes (list): a list of attributes
        best_attr_num (int): an attribute number
        fig_size (tuple): figure size

    Returns:
        None
    """

    feature_importances = model.feature_importances_

    attr_importance = pd.DataFrame({'Attribute': attributes, 'Importance': feature_importances})
    attr_importance.sort_values(by=['Importance'], ascending=False, inplace=True)
    print(attr_importance.head())

    print('The plot shows that \'{}\' attribute is the most important feature selected by GradientBoostingClassifier.'.format(attr_importance.iloc[0]['Attribute']))

    fig = plt.figure(figsize=fig_size)
    plt.bar(range(best_attr_num), attr_importance['Importance'][:best_attr_num], align='center', color='purple')
    plt.xticks(range(best_attr_num), attr_importance['Attribute'][:best_attr_num], rotation='vertical')
    plt.title('Importance Distribution According to Best GradientBoostingClassifier Model')
    plt.xlabel('Attribute Name')
    plt.ylabel('Percentage')
    plt.show()


def apply_pca_k_means(azdias_df, customers_df, pca_components, optimum_k):
    """ Applies pca and k-means clustering pipeline to given dataframes

    Args:
        azdias_df (dataframe): azdias dataframe
        customers_df (dataframe): customers dataframe
        pca_components (int): number of principal components
        optimum_k (int): optimum k value

    Returns:
        dataframe: a dataframe that holds percentage distribution of clusters for azdias and customers data
        pipeline: pca and k-means clustering pipeline
    """

    # Define pipeline
    pipe_pca_kmeans = Pipeline([
        ('pca', PCA(n_components=pca_components, svd_solver='auto', random_state=45)),
        ('k_means', KMeans(n_clusters=optimum_k, random_state=45))
    ])

    # Fit pipe_pca_kmeans pipeline
    pipe_pca_kmeans.fit(azdias_df)

    # Create reduced and clustered data for both azdias and customers dataframes
    azdias_predicted = pd.DataFrame(pipe_pca_kmeans.predict(azdias_df), columns=['Cluster_Id'])
    customers_predicted = pd.DataFrame(pipe_pca_kmeans.predict(customers_df), columns=['Cluster_Id'])

    azdias_clusters = azdias_predicted.value_counts().sort_index()
    customers_cluster = customers_predicted.value_counts().sort_index()

    print('azdias_predicted.shape: {}'.format(azdias_predicted.shape))
    print('azdias_predicted head')
    print(azdias_predicted.head())

    # Create a dataframe that holds percentage distribution of clusters for azdias and customers data
    clusters_df = pd.concat([azdias_clusters, customers_cluster], axis=1).reset_index()
    clusters_df.columns = ['cl_id', 'azdias_cl', 'customers_cl']

    clusters_df['azdias_perc'] = (clusters_df['azdias_cl'] / clusters_df["azdias_cl"].sum() * 100)
    clusters_df['customers_perc'] = (clusters_df['customers_cl'] / clusters_df["customers_cl"].sum() * 100)

    return clusters_df, pipe_pca_kmeans


def plot_multiple_bar(df, optimum_k):
    """ Plots percentage distribution of clusters for azdias and customers data

    Args:
        df (dataframe): a dataframe
        optimum_k (int): optimal k value

    Returns:
        None
    """

    x_labels = np.arange(optimum_k)

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x_labels - 0.2, df['azdias_perc'], width=0.2, color='purple', align='center', label='azdias')
    ax.bar(x_labels, df['customers_perc'], width=0.2, color='green', align='center', label='customers')
    plt.legend(loc="upper right")

    plt.title('Percentage distribution of clusters for azdias and customers data')
    plt.xlabel('Cluster id')
    plt.ylabel('Percentage')
    plt.show()
