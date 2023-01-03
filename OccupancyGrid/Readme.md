# Occupancy Grid
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OccupancyGrid/resources/ogrid_lidar.gif" width=30% height=30%>

Occupancy grid is a popular environmental map generation/representation technique, initially emerged for indoor robotics applications in static environments. In this implementation, instead of generating a global map of the environment, we use the occupancy grid to represent the immediate vicinity of the robot. This means that once an object is outside of robot's FOV, we do not retain its information.

## Binary Grid (Fresh grid per measurement)
The simplest way to form an occupancy grid is to discretize the scan measurements into grid cells. Then the cell becomes occupied if it contains any sensor measurements (or if # measurements is greater than a threshold).

### 3D lidar to 2D grid
When working with a 3D lidar and generating a 2D occupancy grid, certain pre-processing on the pointcloud needs to be made:
- Removal of the ground points
- Removal of the returns that are higher than vehicle/sensor level
- Smashing Z-dimension of points to grid level (Z=0)

Reducing elevation of all points to 0 introduces a hitch though. The ray-tracing algorithm in grid mapping that initially emerged for 2D-scanners rightfully assumes that rays do not have overlapping paths. However for 3D-lidars, when the elevations are reduced to the same level at 2D, we end up having multiple rays crossing each others path. Depending on the behavior we want, it can be handled in the following ways:

#### Conservative 
This emulates the 3D lidar closer to a 2D scanner, since cells along the 2D-ray behind the 1st occupied cell are left uncertain.
- Sort points from low range to high range
- Apply Bresenham from sensor to point, assigning all cells inbetween as free.
- If an occupied cell is encountered on the way, early stop.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OccupancyGrid/resources/3d_deterministic_close_first_early_stopping.png" width=30% height=30%>  

#### Inclusive
More representative of what 3D lidar actually sees. Cells along the 2D-ray behind the 1st occupied cell are allowed to assign occupancy.
- Sort points from higher range to low range
- For each lidar point:
    - Apply Bresenham from sensor to point, assigning all cells inbetween as free.
    - Allowed to reassign a cell as occupied if it was assigned free earlier.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OccupancyGrid/resources/3d_deterministic_far_first_early_stopping.png" width=30% height=30%>  

## Probabilistic Grid: Combining Measurements
Since the real world sensors have noise, instead of assigning binary states to the cells, representing the grid as a *belief map* where cells store probability is better suited for real world applications. Such belief maps rely on the Bayes theorem to combine the **current measurement** and the **previous belief map**.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OccupancyGrid/resources/belief_update.png" width=30% height=30%>  

In the equation above, Markov assumption allows us to combine all prior beliefs from `t=0` to `t=k-1` to a single belief.

Instead of directly using probabilities in the equation above, log-odd ratio (logit function) which maps `p = [0,1]` to `(-Inf,Inf)` is generally used. This is mainly because multiplication of low probabilities (as they get closer to 0) causing numerical instabilities due to floating point arithmetic.

Vanilla grid mapping algorithm is as follows:
>- l_(t) = l_(t-1) + inverse_sensor_model(measurement_t) - l_0
>- inverse_sensor_model:
>    - For each cell:
>        - find closest sensor ray to cell in bearing angle
>        - if `range_ray > range_cell`, assign free
>        - if `[range_cell - sigma] < range_ray < [range_cell + sigma]`, assign occupied.
>            - Can inflate the bearing vicinity similarly here.
>        - if `range_ray < range_cell`, assign l_0
>- l_0 is the static prior probability, in case any prior knowledge of the map is known.
>    - if no prior knowledge, set to probability 0.5 where `logit(0.5) = 0`

However, searching for the closest ray for each cell in the grid becomes costly. Instead, we use Bresenham's line algorithm to find the grid cells that are in between the sensor and the hit point, as an alternative inverse sensor model.

### Implementation
State of the cells are propagated throughout the measurements. However, unlike the grid mapping where a global map is updated, here we **transform** the local grid from step `k` to `k+1` by the inverse transform of the relative robot motion. The purpose here is, rather than mapping, retaining a short-term memory of the immediate vicinity of the robot. 

>- For each lidar point:
>    - Apply Bresenham from sensor to point, updating each cell status: 
>        - Up till hitpoint: `logit_cell(i) += logit(0.3)`
>        - At hitpoint: `logit_cell(i) += logit(0.9)`
>- Propagate the grid to the next step:
>    - Apply inverse robot transformation to the grid.
>    - Ignore the shifted cells that falls outside of the grid boundaries.
>    - Cells that are left unassigned by old cells are initialized with 0.5 probability.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OccupancyGrid/resources/ogrid_lidar_carla.gif" width=30% height=30%>

> **Note**
> The gray "dent" in front of the vehicle is due to getting no-returns from the lidar along the long corridors, since the walls at the end of the corridor is further than lidar's measurement range.  

## Grid Resolution

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OccupancyGrid/resources/ogrid_cells.png" width=15% height=15%>

For small objects, a couple of hits in the cell can be dominated by high number of other rays passing through that cell for behind objects. Decreasing the grid resolution would allow better capture of small objects with the cost of computation time. 



