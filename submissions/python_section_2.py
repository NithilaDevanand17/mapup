import pandas as pd
import numpy as np
from datetime import time

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): A DataFrame containing columns 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: Distance matrix with cumulative distances and diagonal values set to 0.
    """ 
    ids = pd.concat([df['id_start'], df['id_end']]).unique()
    n = len(ids) 
    distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)
     
    np.fill_diagonal(distance_matrix.values, 0)
     
    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] = row['distance']  # Make it symmetric
     
    for k in ids:
        for i in ids:
            for j in ids:
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] < distance_matrix.at[i, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    
    return distance_matrix



def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): A DataFrame representing the distance matrix.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []
 
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:   
                unrolled_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': df.at[id_start, id_end]
                })
 
    df_unrolled = pd.DataFrame(unrolled_data)
    
    return df_unrolled



def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame with 'id_start', 'id_end', and 'distance' columns.
        reference_id (int): The ID to use as a reference for average distance.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """ 
    reference_distances = df[df['id_start'] == reference_id]['distance']
    
    if reference_distances.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])   

    reference_avg_distance = reference_distances.mean()
 
    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1
 
    average_distances = df.groupby('id_start')['distance'].mean()
    filtered_ids = average_distances[(average_distances >= lower_bound) & (average_distances <= upper_bound)]
 
    result_df = filtered_ids.reset_index()
    result_df.columns = ['id_start', 'average_distance']
 
    result_df = result_df.sort_values(by='id_start')

    return result_df



def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): A DataFrame containing columns 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns for toll rates, excluding the 'distance' column.
    """ 

    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
 
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate
 
    df = df.drop(columns=['distance'])

    return df



def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): A DataFrame containing toll rates for different vehicle types.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns for time-based toll rates.
    """ 

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_intervals = {
        'Weekday': [(time(0, 0), time(10, 0), 0.8),
                    (time(10, 0), time(18, 0), 1.2),
                    (time(18, 0), time(23, 59, 59), 0.8)],
        'Weekend': [(time(0, 0), time(23, 59, 59), 0.7)]
    }
 
    results = []
 
    for _, row in df.iterrows():
        for day in days:
            if day in days[:5]:   
                for start_time, end_time, discount in time_intervals['Weekday']:
                    results.append({
                        'id_start': row['id_start'],
                        'id_end': row['id_end'],
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        **{vehicle: row[vehicle] * discount for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
                    })
            else:   
                start_time, end_time, discount = time_intervals['Weekend'][0]
                results.append({
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **{vehicle: row[vehicle] * discount for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
                })

    return pd.DataFrame(results)
 


# Distance Matrix Calculation
df = pd.read_csv('C:/Users/PANNI/Documents/mapup/MapUp-DA-Assessment-2024/datasets/dataset-2.csv')
distance_matrix = calculate_distance_matrix(df)
print("Question 1: Distance Matrix:")
print(distance_matrix)



# Unroll Distance Matrix
unrolled_df = unroll_distance_matrix(distance_matrix)
print("\nQuestion 2: Unrolled Distance Matrix:")
print(unrolled_df)
 


# Finding IDs within Percentage Threshold
reference_id = unrolled_df['id_start'].iloc[0]
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print("\nQuestion 3: IDs within 10% of Reference ID's Average Distance:")
print(ids_within_threshold)



# Calculate Toll Rate
toll_rate_df = calculate_toll_rate(unrolled_df)
print("\nQuestion 4: Toll Rates:")
print(toll_rate_df)



# Calculate Time-Based Toll Rates
time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rate_df)
print("\nQuestion 5: Time-Based Toll Rates:")
print(time_based_toll_rates_df)




