import pandas as pd
import numpy as np

# Define the coordinates for each port
coords = {
    'Turku': (60.43731, 22.21885),
    'Mariehamn': (60.09245, 19.92623),
    'Stockholm': (59.31803, 18.0928)
}

# Function to calculate the Euclidean distance between two points
def distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

# Load data
data = pd.read_csv('datasets/vikinggrace/vikinggrace.csv')
# Convert 'DATE TIME (UTC)' to datetime format for better handling
data['DATE TIME (UTC)'] = pd.to_datetime(data['DATE TIME (UTC)'])
data.sort_values('DATE TIME (UTC)', inplace=True)

""" """ # Load the trip metadata
trips_df = pd.read_csv('vikinggrace_trips.csv')
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

# List of trips to extract
trips_to_extract = [
    ('Turku', 'Mariehamn'),
    ('Mariehamn', 'Turku'),
    ('Mariehamn', 'Stockholm'),
    ('Stockholm', 'Mariehamn'),
    ('Stockholm', 'Turku'),
    ('Turku', 'Stockholm')
]

# Extract and save data for each trip
for start_port, end_port in trips_to_extract:
    output_filename = f'{start_port.lower()}_{end_port.lower()}.csv'
    extract_and_save_trip_data(trips_df, start_port, end_port, output_filename) 
    
""" # Define proximity threshold
proximity_threshold = 0.05  # This value should be adjusted based on the geographical scale

# Variables for tracking trips
trips = []
current_trip = None
in_trip = False

# Iterate over the dataset
for index, row in data.iterrows():
    dist_to_ports = {port: distance(row['LATITUDE'], row['LONGITUDE'], coords[port][0], coords[port][1]) for port in coords}
    
    # Check proximity to each port and speed conditions
    for port, dist in dist_to_ports.items():
        if dist < proximity_threshold and row['SPEED'] > 0:
            if not in_trip:
                # Update potential start time every time it's within proximity with non-zero speed
                last_valid_start = {'start': port, 'start_time': row['DATE TIME (UTC)']}
                in_trip = True
                
    # Check for docking conditions to finalize trip
    for port, dist in dist_to_ports.items():
        if in_trip and dist < proximity_threshold and row['SPEED'] == 0 and last_valid_start['start'] != port:
            current_trip = last_valid_start
            current_trip['end'] = port
            current_trip['end_time'] = row['DATE TIME (UTC)']
            trips.append(current_trip)
            current_trip = None
            last_valid_start = None
            in_trip = False

# Create DataFrame from recorded trips
trip_df = pd.DataFrame(trips)
# Count trips based on the route
route_counts = trip_df.groupby(['start', 'end']).size().reset_index(name='Count')
# Save trips into csv
trip_df.to_csv('vikinggrace_trips.csv', index=False)

# Display the counts for each route
print(route_counts)
 """