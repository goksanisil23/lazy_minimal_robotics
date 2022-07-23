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

g2o is 1 example of a graph optimization back-end. It exploits the sparse connectivity of the graphs to solve large sparse linear systems.

## Simple Example
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

