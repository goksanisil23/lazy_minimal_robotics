# Graph Optimization with g2o
In multi-variable least-squares optimization problems, the problem is composed of many error terms. However, if these error terms are only treated as residuals and variables, then it can be hard to tell the relationship between those. In many robotics problems, one observation at a time only partly reveals the information about a subset of the states and not all states at once ***(locality)***. In this case, it's crucial to relate specific measurements **to not all but specific set of variables**. 


Graphs serve as a structured formulation of an optimization problem. They highlight the relationships between variables and observations in a system.

There are 2 main components to graphs:
- Vertex (Node): Represents the state variable to optimize/estimated. 
    
    *(What we eventually want to find out)*

- Edge: Represents pairwise observation between the 2 nodes it connects. Mostly means some kind of "measurement", or a constraint between the vertices.
    
    *(What we can already see/get from the system)*

Edges are often associated to the error terms, since the error is defined as the difference between our model's estimate and the measured sample.

Note that graphs ultimately encode an objective function, which is solved through repeated linearization via known non-linear optimization routines (Gauss-Newton, Levenberg-Marquardt). 

g2o is one example of a graph optimization back-end. It exploits the sparse connectivity of the graphs to solve large sparse linear systems.

## Unary Edge Example
Although graph formulation makes more sense when there is locality in the optimization problem, for the sake of simplicity and to show the generalizability of graph formulation, we solve the curve fitting problem with g2o in this example: 

`y = exp(mx+c)`

Note that all the noisy observation samples are relevant for all the state variables we're trying to optimize,  `m` and `c` in this curve fitting case. Therefore we have:
- 1 vertex containing `m` and `c`
- ***Unary*** edges connecting the vertex above to itself.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/GraphOpt/resources/graph_unary.png" width=50% height=50%>

Each sample point we have along the curve corresponds to 1 unary edge. Meaning, any measurement connects all variables (only 1 2D vector in this case).

The core of the implementation can be summarized by the following lines:
```c++
// Define a Vertex class which will hold our optimization variables: 
// 2 coefficients
class CurveFittingVertex : public g2o::BaseVertex<2, Eigen::Vector2d> {
public:
  void oplusImpl(const double *update) override {
    _estimate += Eigen::Vector2d(update);
  }
};

// Define a Unary Edge class with template parameters as:
// Observation dimension, obs. type and connecting vertex type
class CurveFittingEdge
    : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
  CurveFittingEdge(double x_sample) : _x_sample(x_sample) {}

    // Define the residual within the function call () operator (just like in Ceres)
    // in a templated manner for internal automatic differentiation for error Jacobian calculation
  template <typename T> bool operator()(const T *params, T *residual) const {
    residual[0] = T(_measurement) -
                  evaluteModelAtSamplePoint(params[0], params[1], _x_sample);
    return true;
  }

private:
  double _x_sample;

  G2O_MAKE_AUTO_AD_FUNCTIONS // use autodiff
};

/* ... */

// Construct the graph 

// Add the single vertex to the graph
CurveFittingVertex *vertex = new CurveFittingVertex();
vertex->setEstimate(Eigen::Vector2d(m_init, c_init));
vertex->setId(0);
g2o_optimizer.addVertex(vertex);

// Add edges (observations) to the graph
for (int obs_idx = 0; obs_idx < N_SAMPLES; obs_idx++) {
    CurveFittingEdge *edge = new CurveFittingEdge(x_gt.at(obs_idx));
    edge->setId(obs_idx);
    // connect the 0th vertex on the graph to vertex (2nd arg)
    edge->setVertex(0, vertex);
    edge->setMeasurement(y_obs.at(obs_idx));
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    edge->setRobustKernel(new g2o::RobustKernelCauchy());
    g2o_optimizer.addEdge(edge);
}
g2o_optimizer.initializeOptimization();
g2o_optimizer.optimize(N_ITER);

```

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/GraphOpt/resources/g2o_curve_fit.png" width=50% height=50%>

## Binary Edge Example
A more intuitive interpretation of an edge is when it indeed connects 2 vertices. In this example, we have a robot that is moving along a 1-dimensional line. Each discrete step of the robot is 1.0m. However, the noisy odometry measurements (wheel encoder, IMU, etc.) of the robot causes it to measure it's displacement with a Gaussian noise. Furthermore, at each pose the robot can also measure it's distance to a landmark that is half-way on this 1-D line, again with some Gaussian measurement noise. 

Through graph optimization, we would like to minimize the **maximum likelihood** of the robot after all these noisy measurements.


<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/GraphOpt/resources/binary_edge.png" width=50% height=50%>

We create 2 types of vertices here: a landmark vertex and a robot pose vertex. In order to form a residual, we need:
- a measurement
- a belief with the current estimates of the system

Here the measurement is the noisy distance measurements from the current robot pose to the landmark. And the belief is the difference between the current landmark position estimate and the current robot pose estimate.

Within the graph, the difference between the belief and the measurements are jointly minimized with each added edge.

```c++
class Vertex1DPose : public g2o::BaseVertex<1, double>
{
public:
    double project(const double &landmarkBelief)
    {
        // distance from landmark to robot (which is also what sensor measures)
        // here _estimate is the robot pose
        return landmarkBelief - _estimate;
    }
    // ...
};

// Error model template parameters: observation dimension, observation type, connecting vertex type(s)
class OdometryEdgeProjection
    : public g2o::BaseBinaryEdge<1, double, Vertex1DPose, Vertex1DLandmark>
{
public:
    OdometryEdgeProjection() {}

    virtual void computeError() override
    {
        auto v0 = static_cast<Vertex1DPose *>(_vertices[0]);
        auto v1 = static_cast<Vertex1DLandmark *>(_vertices[1]);
        auto projectedLandmark = v0->project(v1->estimate());
        _error[0] = projectedLandmark - _measurement;
    }
    // ...
};
```

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/GraphOpt/resources/1d_g2o.png" width=100% height=50%>