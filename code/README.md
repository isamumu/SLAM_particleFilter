# ECE 276 - PR #2

### How to Run

* make sure that the dataset path relative to the code folder should be "../data/
* make sure that python3 and opencv are installed on the machine (pip install opencv-python opencv-python-headless)
* to run type "python3 <filename>" to the terminal from the code folder in the terminal

### File Descriptions
* deadReck2.py: runs dead reckoning and plots a 2D map of what it would look like
* plot_trajectory.py: runs read reckoning and plots the linear and angular trajectories of the robot
* firstScan.py: runs the first LiDAR scan and generates the corresponding map. Mainly used to demonstrate that LiDAR scanning works
* plot_noisy.py: plot the linear and angular trajectories of N=10 hypotheses
* slam.py: generate the 2D map using particle filtering (currently configured for N=500). 
* texture.py: generate the 2D texture map

#### How to Change datasets 
* deadReck2.py: change the dataset number on line 159
* plot_trajectory.py: change the dataset number on line 189
* firstScan.py: change the dataset number on line 189
* plot_noisy.py: change the dataset number on line 189. Change N value on line 190
* slam.py: change the dataset number on line 189. Change N value on line 229
* texture.py: change the dataset number on line 170

#### NOTE
* particle filtering will take about 15 minutes to run for N=10. Will take at least an hour for N=500
* texture mapping will take about 30 minutes to run due to nested loops
