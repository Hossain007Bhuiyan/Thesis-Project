import pandas as pd
import numpy as np
# Define the coordinates for Helsinki and Tallinn
helsinki_coords = (60.148, 24.91)
tallinn_coords = (59.44, 24.77)
# Function to calculate the Euclidean distance of each of the ports from the mystar
def distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
# Load data
data = pd.read_csv('mystar.csv')
# Convert 'DATE TIME (UTC)' to datetime format for better handling
data['DATE TIME (UTC)'] = pd.to_datetime(data['DATE TIME (UTC)'])
data.sort_values('DATE TIME (UTC)', inplace=True)

# Define proximity threshold
# Load the trip metadata
trips_df = pd.read_csv('mystar_trips.csv')
trips_df['start_time'] = pd.to_datetime(trips_df['start_time'])
trips_df['end_time'] = pd.to_datetime(trips_df['end_time'])

# Function to extract and save data for specified trips
def extract_and_save_trip_data(trips_df, start_port, end_port, output_filename):
    filtered_trips = trips_df[(trips_df['start'] == start_port) & (trips_df['end'] == end_port)]
    all_trip_data = pd.DataFrame()

    for trip_id, trip in enumerate(filtered_trips.itertuples(), start=1):
        # Extract data points for the current trip
        trip_data = data[(data['DATE TIME (UTC)'] >= trip.start_time) & (data['DATE TIME (UTC)'] <= trip.end_time)].copy()
        trip_data['tripID'] = trip_id  # Assign the trip ID
        all_trip_data = pd.concat([all_trip_data, trip_data], ignore_index=True)

    # Save the trip data to a CSV file
    all_trip_data.to_csv(output_filename, index=False)
    print(f"Data for trips from {start_port} to {end_port} has been saved to {output_filename}.")

# Extract and save data for Helsinki to Tallinn trips
extract_and_save_trip_data(trips_df, 'Helsinki', 'Tallinn', 'hel_tal.csv')

# Extract and save data for Tallinn to Helsinki trips
extract_and_save_trip_data(trips_df, 'Tallinn', 'Helsinki', 'tal_hel.csv')

# Define proximity threshold
proximity_threshold = 0.05  

# Variables for tracking trips
trips = []
current_trip = None
in_trip = False
# Iterate over the dataset
for index, row in data.iterrows():
    dist_to_helsinki = distance(row['LATITUDE'], row['LONGITUDE'], helsinki_coords[0], helsinki_coords[1])
    dist_to_tallinn = distance(row['LATITUDE'], row['LONGITUDE'], tallinn_coords[0], tallinn_coords[1])
    # Check proximity to each port and speed conditions
    if dist_to_tallinn < proximity_threshold and row['SPEED'] > 0:
        if not in_trip:
            # Update potential start time every time it's within proximity with non-zero speed
            last_valid_start = {'start': 'Tallinn', 'start_time': row['DATE TIME (UTC)']}
            in_trip = True
    elif dist_to_helsinki < proximity_threshold and row['SPEED'] > 0:
        if not in_trip:
            # Update potential start time every time it's within proximity with non-zero speed
            last_valid_start = {'start': 'Helsinki', 'start_time': row['DATE TIME (UTC)']}
            in_trip = True
    # Check for docking conditions to finalize trip
    if in_trip and ((dist_to_tallinn < proximity_threshold and row['SPEED'] == 0 and last_valid_start['start'] == 'Helsinki') or
                    (dist_to_helsinki < proximity_threshold and row['SPEED'] == 0 and last_valid_start['start'] == 'Tallinn')):
        current_trip = last_valid_start
        current_trip['end'] = 'Tallinn' if last_valid_start['start'] == 'Helsinki' else 'Helsinki'
        current_trip['end_time'] = row['DATE TIME (UTC)']
        trips.append(current_trip)
        current_trip = None
        last_valid_start = None
        in_trip = False
# Create DataFrame from recorded trips
trip_df = pd.DataFrame(trips)
# Count trips based on the route
route_counts = trip_df.groupby(['start', 'end']).size().reset_index(name='Count')
#save trips into csv
trip_df.to_csv('mystar_trips.csv', index=False)

# Display the counts for each route
print(route_counts)