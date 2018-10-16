def print_sample(df, columns_only=False, sample_size=5): # TESTED
    """
    Prints columns of the dataframe with a sample of values in them
    If number of unique values is 10 or less it'll print them all
    :param df: pd.DataFrame information about which we'd like to print
    :param columns_only: bool. If you want to print only column names make this on True
    :param sample_size:
    :return:
    """
    print('Shape: {}'.format(df.shape))
    print()
    for i, column in enumerate(df.columns):
        print(i, column)
        if not columns_only:
            unique = df[column].unique()
            types = df[column].apply(type).unique()
            print('types: {}'.format(types))
            if unique.shape[0] <= 10:
                print(unique.tolist())
            else:
                print('sample from {} unique entries'.format(unique.shape[0]))
                print(np.random.choice(unique,sample_size))
            if sum(pd.isnull(pd.Series(unique))) > 0:
                count = (1 - df[column].count()/df[column].shape[0])*100
                print('{0:.2f}% missing values'.format(count))
            print(' ')


def clean_df_default(df,time_columns=None):
    def remove_spaces(name):
        return re.sub(' ', '', name)

    def camel2snake(name):
        name = re.sub(' ', '_', name)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def lower_all_strings_in_df(df):
        """
        lowers all the strings it finds in a given data frame
        :return: None
        """

        def lower_val(val):  # TESTED
            if type(val) == str:
                return val.lower()
            else:
                return val

        df = df.copy()
        for c in df.columns:
            df[c] = df[c].apply(lambda x: lower_val(x))

        return df

    def convert_time_columns(df, time_columns=None):
        if time_columns:
            df = df.copy()
            df[time_columns] = df[time_columns].apply(pd.to_datetime)
        return df

    return (
        df
        .rename(columns=remove_spaces)
        .rename(columns=camel2snake)
        .pipe(lower_all_strings_in_df)
        .pipe(convert_time_columns,time_columns)
    )


def original_file_to_pandas(path_to_file,file_type='csv',encoding='utf-8',sep=',',
                            dtype=None,time_columns=None,
                            index_col=None): # TESTED
    """
    takes a data dump in the form of excel and
    makes all column names snake styled as well as lowers all the
    strings in the df
    :param: str. path to the excel file
    :file_type: one of: 'excel', 'csv'
    :encoding: str (only for csv) #'iso-8859-1'
    :return: pd.DataFrame
    """
    if file_type == 'excel':
        df = pd.read_excel(path_to_file)
    elif file_type == 'csv':
        df = pd.read_csv(path_to_file,sep=sep,encoding=encoding,dtype=dtype,index_col=index_col)
    else:
        raise ValueError('only excel and csv values are supported')

    return clean_df_default(df,time_columns=time_columns)


def add_column(df,old_col,new_col,func):
    df = df.copy()
    df.loc[:,new_col] = df[old_col].apply(func)
    return df

def substitute_column(df,col,func):
    df = df.copy()
    df.loc[:,col] = df[col].apply(func)
    return df


def series_to_dummies(series):
    df = pd.get_dummies(series)
    df.columns = ['{}_{}'.format(series.name,x) for x in df.columns]
    return df


def value_into_integer(value, substitute_with=0):
    if type(value) == str:
        try:
            return int(''.join(value.split(",")))
        except:
            return substitute_with
    else:
        return substitute_with


def make_integer_columns(df, columns_list, substitute_with=0):
    for c in columns_list:
        df.loc[:, c] = df[c].apply(lambda x: value_into_integer(x, substitute_with=substitute_with))


def make_date_columns(df, columns_list):
    for c in columns_list:
        df.loc[:, c] = pd.to_datetime(df[c])


def remove_mostly_empty_cols(df,max_empty=0.5):
    max_filled = df.shape[0]
    to_delete = []
    for column in df.columns:
        if df[column].count() < max_filled * max_empty:
            to_delete.append(column)
    df.drop(to_delete,axis=1,inplace=True)

def sort_dict_by_values(d):
    return  sorted([(k, v) for (k, v) in d.items()], key=lambda x: x[1], reverse=True)

def counter_df(series,column_names = ('thing','counts'),if_percent=False):
    x = pd.DataFrame(sort_dict_by_values(Counter(series)))
    x.columns = column_names
    if if_percent:
        x.loc[:,'percents'] = x.counts/x.counts.sum()
    return x


def merge_lists(l_of_l):
    return [item for sublist in l_of_l for item in sublist]
