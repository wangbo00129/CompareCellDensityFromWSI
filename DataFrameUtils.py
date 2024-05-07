from functools import reduce

def filterSpecificColumnsByQuantile(df_in, column_quantiles={
    'Percentage of invasive tumor':(0.1,0.9),
    'Percentage of tumor-associated stroma':(0.1,0.9)
}):
    df_in = df_in.copy()
    print(df_in.shape)
    filters = []
    for c,(low,high) in column_quantiles.items():
        filter_for_c = (df_in[c] > df_in[c].quantile(low)) & (df_in[c] < df_in[c].quantile(high))
        filters.append(filter_for_c)
    
    df_filtered = df_in[reduce(lambda x,y:x&y, filters)].copy()
    print(df_filtered.shape)
    return df_filtered
