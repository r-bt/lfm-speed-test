import matplotlib.pyplot as plt
import glob
import json

folders = glob.glob("results/*")

data = []

for folder in folders:

    data.append({
        'model': folder.split("/")[-1],
        'results': []
    })

    files = glob.glob(f"{folder}/*.json")

    for file in files:
        
        with open(file, "r") as f:
            result = json.load(f)

            avg_time = result['average_time']
            average_predicted_per_second = result['average_predicted_per_second']

            data[-1]['results'].append({
                'name': file.split("/")[-1].split(".")[0],
                'avg_time': avg_time,
                'average_predicted_per_second': average_predicted_per_second
            })

# In data we have different models, each with a list of results, each result has a name
# For each model we want to plot a bar chart of the average time taken for each result
# Broken down by the name of the result 

import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(data), 2, figsize=(8, 4 * len(data)))

# Ensure axes is iterable when there's only one model
if len(data) == 1:
    axes = [axes]

for i, model_data in enumerate(data):
    names = [result['name'] for result in model_data['results']]
    avg_times = [result['avg_time'] for result in model_data['results']]
    avg_predicted_per_second = [result['average_predicted_per_second'] for result in model_data['results']]

    ax_time = axes[i][0]
    ax_pps = axes[i][1]

    ax_time.bar(names, avg_times)
    ax_time.set_title(f"Model: {model_data['model']}")
    ax_time.set_xlabel('Result Name')
    ax_time.set_ylabel('Average Time')
    ax_time.tick_params(axis='x', rotation=45)

    ax_pps.bar(names, avg_predicted_per_second)
    ax_pps.set_title(f"Model: {model_data['model']}")
    ax_pps.set_xlabel('Result Name')
    ax_pps.set_ylabel('Average Predicted Per Second')
    ax_pps.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()




    

