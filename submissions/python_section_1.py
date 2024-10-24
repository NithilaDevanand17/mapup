from typing import Dict, List
import pandas as pd
import re
import polyline
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta 
from datetime import time
   

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    length = len(lst)

    for i in range(0, length, n): 
        left, right = i, min(i + n - 1, length - 1)
        while left < right:
            lst[left], lst[right] = lst[right], lst[left]
            left += 1
            right -= 1

    return lst

print("Question 1: Reverse List by N Elements")
lst = [1, 2, 3, 4, 5, 6, 7, 8]
n = 3
print(reverse_by_n_elements(lst, n))

 

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    
    for word in lst:
        length = len(word)
        if length not in result:
            result[length] = []
        result[length].append(word)
    #return dict 
    return dict(sorted(result.items())) 

print("\nQuestion 2: Lists & Dictionaries")
lst = ["apple", "bat", "car", "elephant", "dog", "bear"]
print(group_by_length(lst))



def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten_helper(d, parent_key=''):
        items = []
        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(flatten_helper(value, new_key).items())
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        items.extend(flatten_helper(item, f"{new_key}[{i}]").items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, value))
        #return dict
        return dict(items)
    
    if not nested_dict:
        return {}
        
    return flatten_helper(nested_dict)

print("\nQuestion 3: Flatten a Nested Dictionary")
nested = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}
result = flatten_dict(nested)
print(result)



def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """ 
    def backtrack(counter, curr, path):
        if len(path) == len(nums):
            result.append(path[:])
            return
            
        for num in counter:
            if counter[num] > 0:
                counter[num] -= 1
                path.append(num)
                backtrack(counter, curr + 1, path)
                path.pop()
                counter[num] += 1
    
    counter = {}
    for num in nums:
        counter[num] = counter.get(num, 0) + 1
    
    result = []
    backtrack(counter, 0, [])
    return result
    pass

print("\nQuestion 4: Generate Unique Permutations")
print(unique_permutations([1, 1, 2]))


 

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    
    patterns = [
        r"(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-([12][0-9]{3})",   
        r"(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/([12][0-9]{3})",   
        r"([12][0-9]{3})\.(0[1-9]|1[0-2])\.(0[1-9]|[12][0-9]|3[01])"  
    ]
    
    result = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            result.append(match.group())
    
    return result
    pass 

print("\nQuestion 5: Find All Dates in a Text")
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
dates = find_all_dates(text)
print(dates)



def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    
    def haversine_distance(lat1, lon1, lat2, lon2): 
        R = 6371000   
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
         
        dlat = lat2 - lat1
        dlon = lon2 - lon1
         
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
     
    coordinates = polyline.decode(polyline_str)
     
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
     
    distances = [0]   
    
    for i in range(1, len(df)):
        prev_lat = df.iloc[i-1]['latitude']
        prev_lon = df.iloc[i-1]['longitude']
        curr_lat = df.iloc[i]['latitude']
        curr_lon = df.iloc[i]['longitude']
        
        distance = haversine_distance(prev_lat, prev_lon, curr_lat, curr_lon)
        distances.append(distance)
    
    df['distance'] = distances 
    #return pd.DataFrame()
    return pd.DataFrame(df)

print("\nQuestion 6: Decode Polyline, Convert to DataFrame with Distances")
polyline_str = "_gv~Ffv{uOwBfIyBdIcCvHmCnHwCdH"
df = polyline_to_dataframe(polyline_str)
print(df)

 


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    
    n = len(matrix) 
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)] 
    final_matrix = [[0] * n for _ in range(n)]  
    for i in range(n):
        for j in range(n): 
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    #return []
    return final_matrix   

print("\nQuestion 7: Matrix Rotation and Transformation")
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
] 
result = rotate_and_multiply_matrix(matrix)
for row in result:
    print(row)




def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of the time data by checking whether the timestamps 
    for each unique (id, id_2) pair cover a full 24-hour period and span all 7 days.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pd.Series: A boolean series indicating if each (id, id_2) pair has incorrect timestamps.
    """ 

    all_days = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
     
    results = pd.Series(index=pd.MultiIndex.from_product([df['id'].unique(), df['id_2'].unique()]), dtype=bool)
 
    grouped = df.groupby(['id', 'id_2'])

    for (id_value, id_2_value), group in grouped: 
        present_days = set(group['startDay'].unique()).union(set(group['endDay'].unique()))
 
        if present_days != all_days:
            results.loc[(id_value, id_2_value)] = True   
            continue
 
        time_coverage = {}
        for _, row in group.iterrows():
            day = row['startDay']
            start_time = pd.to_datetime(row['startTime'], format='%H:%M:%S').time()
            end_time = pd.to_datetime(row['endTime'], format='%H:%M:%S').time()
 
            if day not in time_coverage:
                time_coverage[day] = []

            time_coverage[day].append((start_time, end_time))
 
        all_full_coverage = True
        for day, times in time_coverage.items():
            combined_times = []
            for start, end in times:
                combined_times.append((pd.to_datetime(start.strftime('%H:%M:%S')), pd.to_datetime(end.strftime('%H:%M:%S'))))
             
            combined_intervals = []
            for start, end in combined_times:
                combined_intervals.append((start.time(), end.time())) 
            if min(start for start, end in combined_intervals) > time(0, 0) or max(end for start, end in combined_intervals) < time(23, 59, 59):
                all_full_coverage = False
                break
        
        results.loc[(id_value, id_2_value)] = not all_full_coverage

    return results

print("\nQuestion 8: Time Check")
df = pd.read_csv('C:/Users/PANNI/Documents/mapup/MapUp-DA-Assessment-2024/datasets/dataset-1.csv')
result = time_check(df)
print(result)

