# Occupancy Grid
- Grid is defined by lenght, width, resolution
- define where vehicle's origin is w.r.t grid origin


## Static occupancy grid
- occupancy of the cells are represented by the log-odd ratios, instead of directly representing the probabilities. This is mainly because multiplication low probabilities (as they get closer to 0) causes numerical instabilities due to floating point arithmetic.
Log-odd function: log(prob/1-prob) --> (-inf,inf)
    - l_(t) = l_(t-1) + inverse_sensor_model(measurement_t) - l_0
    - inverse_sensor_model:
        - constant low probability (0.2) before (range-sigma)
        - gaussian around [(range-sigma),(range+sigma)], reaching 0.8 max
        - constant even probability (0.5), after (range+sigma)
        where sigma is the variance of our sensor range measurement uncertainty
    - l_0 is the static prior probability, in case any prior knowledge of the map is known.
        - if no prior knowledge, set to probability 0.5 

Instead of making this update for every single cell in the grid, use Bresenham's line algorithm to find the gric cells that are in the path of the ray emitted by the lidar.

## Dynamic Occupancy Grid
- Each grid cell is classified dynamic or static.
- use dynamic grid cells for object level tracking hypothesis
- each dynamic grid cell is considered for being assigned to existing tracks.
    - successfull assignment if (-)log likelihood between grid cell and track is below threshold.
    - if not, unassigned cells lead to new tracks.
        - since multiple unassigned cells might belong to same track, DBSCAN clustering is used.
- to reduce false positives of dynamic cells to tracks:
    - new tracks can only be created if sufficient number of cells contribute in dbscan.
    - a new track remains tentative until it is detected M out of N times. 
- have a deletion threshold for the tracks.
    - allow the track to coast away for a couple of steps, and let it reassociate the grid cells if they can recover later.

- the velocity and heading states of the track are obtained via the average velocity distribution of the associated grid cells.
    - length, width, orientation of the tracked object is also obtained using the spatial distribution of the associated grid cells.