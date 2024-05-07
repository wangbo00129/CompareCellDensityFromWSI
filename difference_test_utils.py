
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
from scikit_posthocs import posthoc_dunn
import statsmodels.api as sm

def test_continuous_by_discrete(cont_series: pd.Series, disc_series: pd.Series):
    """
    对连续变量按离散变量分组进行统计检验。

    参数:
    cont_series: pd.Series
        连续变量的数据。
    disc_series: pd.Series
        离散变量的数据，预期为分类变量。

    返回:
    results: dict
        包含检验结果的字典。
    """
    # 检查两个Series的长度是否相同
    if len(cont_series) != len(disc_series):
        raise ValueError("The length of continuous and discrete series must be the same.")

    # Mann-Whitney U检验，适用于两组比较
    def mann_whitney(cont1, cont2, disc1, disc2):
        stat, p_value = mannwhitneyu(cont1, cont2)
        return {'stat': stat, 'p_value': p_value, 'test': 'Mann-Whitney U Test'}

    # Kruskal-Wallis H检验，适用于多组比较
    def kruskal_wallis(cont_values, disc_values):
        stat, p_value = kruskal(cont_values, disc_values)
        if p_value < 0.05:
            posthoc_results = posthoc_dunn(pd.DataFrame({'Continuous':cont_values, 'Discrete':disc_values}), val_col='Continuous', group_col='Discrete', p_adjust='holm', sort=True)
            return {'stat': stat, 'p_value': p_value, 'test': 'Kruskal-Wallis H Test', 'posthoc': posthoc_results}
        else:
            return {'stat': stat, 'p_value': p_value, 'test': 'Kruskal-Wallis H Test'}

    disc_series_encoded = pd.Categorical(disc_series)
    disc_series_encoded = disc_series_encoded.codes
    disc_series = disc_series_encoded

    # 根据离散变量的类别数决定使用哪种检验
    if len(set(disc_series)) == 2:
        # 如果只有两个组，使用Mann-Whitney U检验
        group1 = cont_series[disc_series == disc_series.unique()[0]]
        group2 = cont_series[disc_series == disc_series.unique()[1]]
        return mann_whitney(group1, group2, disc_series.unique()[0], disc_series.unique()[1])
    else:
        # 如果有多于两个组，使用Kruskal-Wallis H检验
        return kruskal_wallis(cont_series, disc_series)

# 示例使用
# continuous_data = pd.Series([...])  # 连续变量数据
# discrete_data = pd.Series([...])  # 离散变量数据
# results = test_continuous_by_discrete(continuous_data, discrete_data)

def significant_difference_test_for_dataframe(df: pd.DataFrame, discrete_vars: list, continuous_vars: list):
    """
    对DataFrame中的每一对离散变量和连续变量进行统计检验。

    参数:
    df: pd.DataFrame
        包含数据的DataFrame。
    discrete_vars: list
        离散变量的列名列表。
    continuous_vars: list
        连续变量的列名列表。

    返回:
    overall_results: dict
        包含每一对变量检验结果的字典。
    """
    overall_results = {}
    for cont_var in continuous_vars:
        results = {}
        for disc_var in discrete_vars:
            # 提取对应的连续变量和离散变量的Series
            print(cont_var, disc_var)
            cont_series = df[cont_var]
            disc_series = df[disc_var]
            
            # 对连续变量按离散变量分组进行统计检验
            results[disc_var] = test_continuous_by_discrete(cont_series, disc_series)
        
        overall_results[cont_var] = results

    return overall_results

if __name__ == '__main__':
    # 示例DataFrame
    df = pd.DataFrame({
        'discrete_var1': ['a','b','c','b'],
        'discrete_var2': ['e','e','g','g'],
        'continuous_var1': [1,2,5,6],
        'continuous_var2': [2,3,5,6]
    })

    # 结果
    results = significant_difference_test_for_dataframe(df, ['discrete_var1', 'discrete_var2'], ['continuous_var1', 'continuous_var2'])
    print(pd.DataFrame(results))
    # test_continuous_by_discrete(df['continuous_var1'], df['discrete_var1'])