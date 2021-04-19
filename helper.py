import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pickle


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


def drop_unnecessary_cols(df, attr_by_missing_info, drop_percentage=0.5):
    """ Finds and drops columns that have more than drop_percentage nan values

    Args:
        df (dataframe): a dataframe
        attr_by_missing_info (dataframe): dataframe holds attribute and their nan percentages
        drop_percentage (float): percentage

    Returns:
        None
    """

    attr_dropped = list(attr_by_missing_info[attr_by_missing_info['Null Percentage'] > drop_percentage]['Attribute'])
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


def save_dataframe(df, file_name):
    """ Saves dataframe into a file

    Args:
        df (dataframe): a dataframe
        file_name (string): name of the file

    Returns:
        None
    """

    df.to_pickle(file_name)


def load_dataframe(file_name):
    """ Loads dataframe from a file

    Args:
        file_name (string): name of the file

    Returns:
        dataframe: a dataframe is loaded from a file
    """

    df = pd.read_pickle(file_name)
    return df
