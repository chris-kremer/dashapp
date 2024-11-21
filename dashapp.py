# Import necessary libraries
import os
import pandas as pd
from datetime import date, timedelta, datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pytz
import streamlit as st

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the main function
def generate_plot():
    # File paths and configurations
# File paths and configurations
    repo_path = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
    filename = os.path.join(repo_path, 'Data_10.04.24.xlsx')  # Excel file path in the same repo
    encoding_type = 'ISO-8859-1'
    sheets = ["sleep", "free_time", "schedule", "caffein"]
    dfs = []

    # Define local timezone
    local_tz = pytz.timezone('Europe/Berlin')  # Replace with your local timezone

    # Load and process each sheet
    for sheet in sheets:
        df = pd.read_excel(filename, sheet_name=sheet)
        if sheet == "sleep":
            df = df.drop(columns=["Schlafzeit", "Wecker", "Überschlafen", "Sleep_Stop_Plan"], errors='ignore')
        elif sheet == "free_time":
            df = df.drop(columns=["length", "Time"], errors='ignore')
        elif sheet == "schedule":
            df = df.drop(columns=["Comment", "Delay", "one", "two", "three"], errors='ignore')
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna(subset=['Date'])

    # Read the 'schedule' sheet separately for the top to-dos
    df_schedule = pd.read_excel(filename, sheet_name='schedule')
    df_schedule = df_schedule.drop(columns=["Comment", "Delay", "one", "two", "three"], errors='ignore')

    # Ensure 'duration' is numeric
    df_schedule['duration'] = pd.to_numeric(df_schedule['duration'], errors='coerce').fillna(0)

    # Ensure 'Date' is datetime
    df_schedule['Date'] = pd.to_datetime(df_schedule['Date'], errors='coerce')

    # Filter tasks scheduled for today and not done (duration == 0)
    df_schedule_not_done = df_schedule[
        (df_schedule['Date'].dt.date == date.today()) &
        (df_schedule['duration'] == 0)
    ]

    # Ensure 'Estimate' and 'Priority' are numeric
    df_schedule_not_done['Estimate'] = pd.to_numeric(df_schedule_not_done['Estimate'], errors='coerce').fillna(0)
    df_schedule_not_done['Priority'] = pd.to_numeric(df_schedule_not_done['Priority'], errors='coerce').fillna(0)

    # Compute 'Adj. Priority'
    df_schedule_not_done['Adj. Priority'] = df_schedule_not_done.apply(
        lambda x: (x['Priority'] / x['Estimate'] * 60) if x['Estimate'] != 0 else 0, axis=1)

    # Sort and select top 7
    top_todos = df_schedule_not_done.sort_values(by='Adj. Priority', ascending=False).head(7)

    # Load sleep data from the CSV
    #filename_sleep = os.path.join(desktop_path, 'Desktop', 'Dektop', 'Sleep Analysis Data 3.csv')
    #df_sleep = pd.read_csv(filename_sleep, encoding=encoding_type, delimiter=',', on_bad_lines='skip')

    # Process df_sleep
    #one_week_ago = pd.Timestamp.now(tz='UTC') - pd.Timedelta(weeks=1)
    #df_sleep['Start'] = pd.to_datetime(df_sleep['Start'], utc=True)
    #df_sleep['End'] = pd.to_datetime(df_sleep['End'], utc=True)
    #df_sleep = df_sleep[df_sleep['Start'] >= one_week_ago]
    #df_sleep.rename(columns={'End': 'stop', 'Start': 'start', 'Value': 'Activity'}, inplace=True)
    #df_sleep = df_sleep[df_sleep['Source'] == 'AppleÂ Watch von Christian']
    #df_sleep['Priority'] = 0
    #df_sleep['Category'] = 'sleep'
    #df_sleep['Comment'] = ''
    #df_sleep['Adj. Priority'] = 0
    #df_sleep['Estimate'] = 60
    #df_sleep['duration'] = (df_sleep['stop'] - df_sleep['start']).dt.total_seconds() / 60

    # Drop unnecessary columns
    #df_sleep.drop(columns=['Duration (hr)', 'Source'], inplace=True, errors='ignore')
    #df_sleep['Date'] = df_sleep['start'].dt.date
    #df_sleep = df_sleep[~df_sleep['Activity'].isin(['InBed', 'Asleep'])]

    # File paths for calendars
    #file_path_schule = os.path.join(desktop_path, 'Desktop', 'Dektop', 'Calenders', 'Schule_Events.csv')
    #file_path_arbeit = os.path.join(desktop_path, 'Desktop', 'Dektop', 'Calenders', 'Arbeit_Events.csv')
    #file_path_dashboard = os.path.join(desktop_path, 'Desktop', 'Dektop', 'Calenders', 'Dashboard_Events.csv')

    # Read calendar CSV files
    #df_schule = pd.read_csv(file_path_schule, encoding=encoding_type, delimiter=',', on_bad_lines='skip')
    #df_arbeit = pd.read_csv(file_path_arbeit, encoding=encoding_type, delimiter=',', on_bad_lines='skip')
    #df_dashboard = pd.read_csv(file_path_dashboard, encoding=encoding_type, delimiter=',', on_bad_lines='skip')

    # Assign additional attributes
    #df_schule['type'] = 'Schule'
    #df_arbeit['type'] = 'Arbeit'
    #df_dashboard['type'] = 'Dashboard'
    #df_schule['Priority'] = 5
    #df_arbeit['Priority'] = 4
    #df_dashboard['Priority'] = 3

    # Combine calendar data
    #df_cals = pd.concat([df_schule, df_arbeit, df_dashboard], ignore_index=True)

    # Specify the correct format for your date columns
    date_format = "%d.%m.%y %H:%M:%S"

    # Convert 'Start Date' and 'End Date' columns to datetime, specifying the correct format
    #df_cals['Start Date'] = pd.to_datetime(df_cals['Start Date'], format=date_format, errors='coerce', dayfirst=True)
    #df_cals['End Date'] = pd.to_datetime(df_cals['End Date'], format=date_format, errors='coerce', dayfirst=True)

    # Remove rows where date conversion failed
    #df_cals = df_cals.dropna(subset=['Start Date', 'End Date'])

    # Rename columns for further processing
    #df_cals.rename(columns={'Summary': 'Activity', 'Start Date': 'start', 'End Date': 'stop'}, inplace=True)

    # Further processing for plotting
    #df_cals['Estimate'] = (df_cals['stop'] - df_cals['start']).dt.total_seconds() / 60
    #df_cals['Adj. Priority'] = df_cals.apply(
        #lambda x: (x['Priority'] / x['Estimate'] * 60) if x['Estimate'] != 0 else 0, axis=1)
    #df_cals['Date'] = df_cals['start'].dt.date

    # Combine everything
    combined_df_e_c = combined_df
    combined_df_e_c['Date'] = pd.to_datetime(combined_df_e_c['Date'], errors='coerce')

    # Ensure 'start' and 'stop' columns are in datetime format and localized
    combined_df_e_c['start'] = pd.to_datetime(combined_df_e_c['start'], errors='coerce')
    combined_df_e_c['start'] = combined_df_e_c['start'].dt.tz_localize(local_tz, ambiguous='NaT', nonexistent='NaT')
    combined_df_e_c['start'] = combined_df_e_c['start'].dt.tz_convert(pytz.utc)

    combined_df_e_c['stop'] = pd.to_datetime(combined_df_e_c['stop'], errors='coerce')
    combined_df_e_c['stop'] = combined_df_e_c['stop'].dt.tz_localize(local_tz, ambiguous='NaT', nonexistent='NaT')
    combined_df_e_c['stop'] = combined_df_e_c['stop'].dt.tz_convert(pytz.utc)

    # Shift every event by +1 hour
    combined_df_e_c['start'] = combined_df_e_c['start'] + timedelta(hours=1)
    combined_df_e_c['stop'] = combined_df_e_c['stop'] + timedelta(hours=1)

    # Ensure 'duration' column exists and is numeric
    combined_df_e_c['duration'] = pd.to_numeric(combined_df_e_c['duration'], errors='coerce').fillna(0)

    # Filter for today
    today = date.today()
    df_today = combined_df_e_c[
        (combined_df_e_c['start'].dt.date <= today) &
        (combined_df_e_c['stop'].dt.date >= today)
    ]

    # Filter today's tasks into done and not done
    df_today_done = df_today[df_today['duration'] > 0]
    df_today_not_done = df_today[df_today['duration'] <= 0]

    # Ensure necessary columns exist in plot_today
    required_columns = ['Priority', 'duration', 'start', 'stop', 'Activity', 'color', 'Category', 'done', 'Estimate', 'Adj. Priority']
    for col in required_columns:
        if col not in df_today_done.columns:
            df_today_done[col] = np.nan
        if col not in df_today_not_done.columns:
            df_today_not_done[col] = np.nan

    # Ensure 'Estimate' and 'Priority' are numeric
    df_today_not_done['Estimate'] = pd.to_numeric(df_today_not_done['Estimate'], errors='coerce').fillna(0)
    df_today_not_done['Priority'] = pd.to_numeric(df_today_not_done['Priority'], errors='coerce').fillna(0)

    # Compute 'Adj. Priority' for df_today_not_done
    df_today_not_done['Adj. Priority'] = df_today_not_done.apply(
        lambda x: (x['Priority'] / x['Estimate'] * 60) if x['Estimate'] != 0 else 0, axis=1)

    # Define the schedule_tasks function
    def schedule_tasks(df_events, df_tasks):
        now = datetime.now(local_tz).astimezone(pytz.utc)  # Current time in UTC
        now = now + timedelta(hours=1)
        today = now.date()

        # Ensure 'start' and 'stop' in df_events are timezone-aware (UTC)
        df_events['start'] = pd.to_datetime(df_events['start'], utc=True)
        df_events['stop'] = pd.to_datetime(df_events['stop'], utc=True)

        # Filter future events occurring today and after the current time
        future_events = df_events[(df_events['start'].dt.date == today) & (df_events['start'] > now)]

        scheduled_tasks = []
        occupied_slots = []

        # Track all occupied time slots
        for _, event in future_events.iterrows():
            occupied_slots.append((event['start'], event['stop']))

        # Sort tasks by 'Adj. Priority' in descending order (higher priority first)
        df_tasks = df_tasks.sort_values(by='Adj. Priority', ascending=False)

        for _, task in df_tasks.iterrows():
            start_time = now
            end_time = start_time + timedelta(minutes=task['Estimate'])

            # Check for conflicts and adjust start_time accordingly
            conflict = True
            while conflict:
                conflict = False
                for occupied_start, occupied_stop in occupied_slots:
                    # Check if the current task overlaps with any occupied slot (+10 minutes buffer)
                    if (start_time < occupied_stop + timedelta(minutes=10)) and (end_time > occupied_start - timedelta(minutes=10)):
                        # Conflict detected, shift the start_time
                        start_time = occupied_stop + timedelta(minutes=10)
                        end_time = start_time + timedelta(minutes=task['Estimate'])
                        conflict = True
                        break  # Re-check with updated start_time

            # Create the task's scheduled entry
            scheduled_task = task.to_dict()
            scheduled_task['start'] = start_time
            scheduled_task['stop'] = end_time
            scheduled_task['color'] = 'yellow'
            scheduled_task['type'] = 'auto_planned'
            scheduled_task['duration'] = scheduled_task['Estimate']

            # Append the task to the list of scheduled tasks
            scheduled_tasks.append(scheduled_task)

            # Add the task to occupied slots
            occupied_slots.append((start_time, end_time))

        return pd.DataFrame(scheduled_tasks)

    # Schedule today's not done tasks
    scheduled_tasks_df = schedule_tasks(df_today, df_today_not_done)

    # Processing for plotting
    plot_today = pd.concat([df_today_done, scheduled_tasks_df], ignore_index=True)

    # Ensure necessary columns exist in plot_today
    for col in required_columns:
        if col not in plot_today.columns:
            plot_today[col] = np.nan

    # Fill missing values with defaults
    plot_today['Priority'] = plot_today['Priority'].fillna(0)
    plot_today['duration'] = plot_today['duration'].fillna(0)
    plot_today['Category'] = plot_today['Category'].fillna('')
    plot_today['done'] = plot_today['done'].fillna(0)
    plot_today['Estimate'] = plot_today['Estimate'].fillna(0)
    plot_today['Adj. Priority'] = plot_today['Adj. Priority'].fillna(0)

    # Ensure 'start' and 'stop' are datetime and in UTC
    plot_today['start'] = pd.to_datetime(plot_today['start'], utc=True)
    plot_today['stop'] = pd.to_datetime(plot_today['stop'], utc=True)

    # Get current time in local timezone and convert to UTC
    now = datetime.now(local_tz).astimezone(pytz.utc)
    now = now + timedelta(hours=1)

    # Assign base colors according to specifications
    def assign_base_color(row):
        if row['Activity'].lower() == 'sleep' or row['Category'].lower() == 'sleep':
            return 'blue'
        elif row['Category'].lower() == 'caffein':
            if row['duration'] > 0:
                return 'grey'
            else:
                return 'grey'  # Will be overridden if ongoing
        else:
            if row['Priority'] < 0:
                return 'red'
            elif row['Priority'] == 6:
                return 'grey'
            elif row['Priority'] > 0:
                return 'green'
            elif row['Priority'] == 0:
                return 'white'
            else:
                return 'grey'

    plot_today['base_color'] = plot_today.apply(assign_base_color, axis=1)

    # Initialize 'color' with base color
    plot_today['color'] = plot_today['base_color']

    # Assign yellow color to future events
    plot_today.loc[plot_today['start'] > now, 'color'] = 'yellow'

    # Define mapping of base colors to lighter colors
    light_colors = {
        'blue': 'lightblue',
        'grey': 'lightgrey',
        'red': 'lightcoral',
        'green': 'lightgreen',
        'white': 'lightgrey',
    }

    # Identify events that are currently ongoing
    is_current = (plot_today['start'] <= now) & (plot_today['stop'] >= now)

    # Assign lighter colors to ongoing events based on category
    # Sleep events remain blue even if ongoing
    plot_today.loc[is_current & (plot_today['Activity'].str.lower() == 'sleep'), 'color'] = 'blue'
    plot_today.loc[is_current & (plot_today['Category'].str.lower() == 'caffein'), 'color'] = 'lightgrey'
    plot_today.loc[is_current & ~(plot_today['Category'].str.lower().isin(['caffein', 'sleep'])), 'color'] = plot_today.loc[is_current & ~(plot_today['Category'].str.lower().isin(['caffein', 'sleep'])), 'base_color'].map(light_colors)

    one_hour_from_now = datetime.now() + timedelta(hours=1)

    # Iterate through rows and apply conditions
    plot_today['stop'] = plot_today.apply(
        lambda row: one_hour_from_now if row['duration'] < 0 else row['stop'], axis=1
    )

    # Update 'duration' column based on the difference between 'stop' and 'start'
    plot_today['duration'] = (plot_today['stop'] - plot_today['start']).dt.total_seconds() / 60  # if in minutes


    # Change color to light green for non-sleep categories

    # Define the plotting function
    def plot_activities_of_a_day(df, date_str):
        # Ensure 'start' and 'stop' are in datetime format
        df['start'] = pd.to_datetime(df['start'], utc=True)
        df['stop'] = pd.to_datetime(df['stop'], utc=True)

        # Convert 'date_str' to date
        try:
            date_obj = datetime.strptime(date_str, '%d.%m.%Y').date()
        except ValueError:
            st.write(f"Incorrect date format for {date_str}. Expected format is 'dd.mm.yyyy'.")
            return

        # Filter data for events overlapping with the specified date
        daily_data = df[
            (df['start'].dt.date <= date_obj) &
            (df['stop'].dt.date >= date_obj)
        ]

        if daily_data.empty:
            st.write(f"No data for {date_str}.")
            return

        # Adjust events to fit within the day
        daily_data['adjusted_start'] = daily_data['start'].apply(lambda x: max(x, pd.Timestamp(date_obj, tz='UTC')))
        daily_data['adjusted_stop'] = daily_data.apply(
            lambda row: min(row['stop'], pd.Timestamp(date_obj + timedelta(days=1), tz='UTC')),
            axis=1
        )
        daily_data['duration'] = (daily_data['adjusted_stop'] - daily_data['adjusted_start']).dt.total_seconds() / 60  # duration in minutes
        daily_data['duration_hours'] = daily_data['duration'] / 60  # duration in hours

        # Prepare plotting space with larger size
        fig, ax = plt.subplots(figsize=(18, 10))

        # Prepare priority levels (-1 to 6)
        priority_levels = np.arange(-1, 7)

        # Mapping for priority labels
        priority_labels = {
            -1: 'Unproductive',
            0: 'Basics',
            1: 'Priority 1',
            2: 'Priority 2',
            3: 'Priority 3',
            4: 'Priority 4',
            5: 'Priority 5',
            6: 'caffein'
        }

        y_positions = range(len(priority_levels))
        y_labels = [priority_labels.get(p, f'Priority {p}') for p in priority_levels]

        # Loop through each priority level and plot activities
        for i, priority in enumerate(priority_levels):
            df_priority = daily_data[daily_data['Priority'] == priority]
            if df_priority.empty:
                continue
            for idx, row in df_priority.iterrows():
                start_time = row['adjusted_start']
                start_hour = (start_time - pd.Timestamp(date_obj, tz='UTC')).total_seconds() / 3600
                duration = row['duration_hours']
                activity = row['Activity']
                color = row.get('color', 'grey')

                if color == 'transparent':
                    facecolor = 'none'
                    edgecolor = 'black'
                else:
                    facecolor = color
                    edgecolor = 'black'

                bar = ax.broken_barh(
                    [(start_hour, duration)],
                    (i - 0.4, 0.8),
                    facecolors=facecolor,
                    edgecolors=edgecolor,
                    linewidth=1
                )

                # Add activity labels with larger font size
                ax.text(
                    start_hour + duration / 2,
                    i,
                    activity,
                    ha='center',
                    va='center',
                    fontsize=12,  # Increased font size
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
                )

        # Set labels and title with larger font size
        ax.set_xlabel('Hour of Day', fontsize=14)
        ax.set_ylabel('Priority Level', fontsize=14)
        ax.set_title(f'Activities on {date_str}', fontsize=16)

        # Set y-axis tick labels font size
        ax.tick_params(axis='y', labelsize=12)
        # Set x-axis tick labels font size
        ax.tick_params(axis='x', labelsize=12)

        # Set y-axis labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)

        # Set limits
        ax.set_xlim(0, 24)
        ax.set_ylim(-1, len(priority_levels))

        # Add vertical line for the current time if it's the same day
        current_date = datetime.now(local_tz).date()
        if date_obj == current_date:
            current_time_local = datetime.now(local_tz)
            current_time_local = current_time_local + timedelta(hours=1)
            current_time_utc = current_time_local.astimezone(pytz.utc)
            current_hour = (current_time_utc - pd.Timestamp(date_obj, tz='UTC')).total_seconds() / 3600
            ax.axvline(x=current_hour, color='grey', linestyle='--', linewidth=1, label='Current Time')

        # Add legend with larger font size
        ax.legend(fontsize=12)

        plt.tight_layout()
        st.pyplot(fig)

    # Plot activities of today
    today_str = date.today().strftime('%d.%m.%Y')
    plot_activities_of_a_day(plot_today, today_str)

    # Display the table of top 7 to-dos with highest adjusted priority

    # Select the columns to display
    columns_to_display = ['Activity', 'Priority', 'Estimate', 'Adj. Priority']

    # Display the table using Streamlit
    st.write("### Top 7 To-Dos with Highest Adjusted Priority for Today")
    st.table(top_todos[columns_to_display])

# Streamlit App Execution

# Generate the plot on initial load
generate_plot()

# Create an Update button
if st.button('Update'):
    generate_plot()