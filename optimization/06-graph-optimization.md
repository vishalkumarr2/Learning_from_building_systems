# 06 — Graph-Based Optimization and SLAM

> **Goal:** Understand how SLAM and multi-sensor fusion are formulated as optimization on graphs — factor graphs, pose graphs, and the solvers (g2o, GTSAM, Ceres) that make them tractable.

> **Second highest OKS value.** The navigation estimator's state estimation and any future SLAM capability depend on graph-based optimization.

---

## 1. From Least Squares to Graphs

### The Key Insight

Many robotics estimation problems have this structure:
- **Variables (nodes):** robot poses, landmark positions, calibration params
- **Measurements (edges):** odometry, sensor readings, loop closures
- Each measurement contributes a **residual** connecting a small subset of variables

This is a **sparse** nonlinear least-squares problem. The sparsity pattern comes from the graph structure.

$$\min_x \sum_{(i,j) \in \mathcal{E}} \|e_{ij}(x_i, x_j)\|^2_{\Omega_{ij}}$$

where:
- $x_i, x_j$ are connected variables
- $e_{ij}$ is the error/residual for edge $(i,j)$
- $\Omega_{ij}$ is the information matrix (inverse covariance) — the weight

---

## 2. Factor Graphs

### Definition

A bipartite graph with two types of nodes:
- **Variable nodes** $x_i$ — the unknowns (robot poses, landmarks)
- **Factor nodes** $f_k$ — the constraints/measurements

Each factor connects to the variables it depends on.

### Example: Robot Driving in a Loop

```
Variable nodes:    x0 --- x1 --- x2 --- x3
                    |                     |
Factor nodes:      f_prior  f_odom  f_odom  f_odom
                                          f_loop_closure (x0-x3)
```

- **Prior factor** $f_0(x_0)$: initial pose (from GPS, startup)
- **Odometry factors** $f_{01}(x_0, x_1)$, ...: wheel encoder measurements
- **Loop closure** $f_{03}(x_0, x_3)$: recognizing a previously visited place

### The Joint Probability View

Each factor represents a probability:

$$p(x_0, x_1, x_2, x_3 | z) \propto f_0(x_0) \cdot f_{01}(x_0, x_1) \cdot f_{12}(x_1, x_2) \cdot f_{23}(x_2, x_3) \cdot f_{03}(x_0, x_3)$$

Maximizing this (MAP estimation) is equivalent to minimizing the sum of squared Mahalanobis errors — which is exactly our graph-based least squares!

$$\min_x \sum_k \|e_k(x)\|^2_{\Omega_k}$$

---

## 3. Pose-Graph SLAM

### The Most Common SLAM Formulation

Variables: robot poses $x_i = (t_i, R_i) \in SE(2)$ or $SE(3)$

Edges:
- **Odometry:** $z_{ij} = $ measured relative pose between $x_i$ and $x_j$
- **Loop closures:** same, but between non-sequential poses

Error for edge $(i,j)$:

$$e_{ij} = \log\left( z_{ij}^{-1} \cdot (x_i^{-1} \cdot x_j) \right)$$

This is the difference between the measured relative pose ($z_{ij}$) and the estimated relative pose ($x_i^{-1} \cdot x_j$), expressed in the Lie algebra.

### Solving the Pose Graph

1. **Linearize:** Compute Jacobians of each $e_{ij}$ w.r.t. $x_i, x_j$
2. **Build normal equations:** $H \Delta x = -b$ where:
   - $H = \sum_{(i,j)} J_{ij}^T \Omega_{ij} J_{ij}$ (approximate Hessian)
   - $b = \sum_{(i,j)} J_{ij}^T \Omega_{ij} e_{ij}$ (gradient)
3. **Solve for $\Delta x$** — exploit sparsity!
4. **Update:** $x \leftarrow x \oplus \Delta x$ (manifold update)
5. **Repeat** until convergence

### Sparsity Pattern

The Hessian $H$ is sparse because each factor only involves a few variables.

For a pose-graph with $n$ poses and $m$ edges:
- $H$ is $3n \times 3n$ (2D) or $6n \times 6n$ (3D)
- But only $O(m)$ non-zero blocks

```
H = [H_00  H_01  0     H_03]     ← pose x0 connects to x1, x3
    [H_10  H_11  H_12  0   ]     ← pose x1 connects to x0, x2
    [0     H_21  H_22  H_23]     ← pose x2 connects to x1, x3
    [H_30  0     H_32  H_33]     ← pose x3 connects to x0, x2
```

**Sparse Cholesky** or **conjugate gradient** (from Ch. 02) solves this efficiently.

---

## 4. Manifold Optimization

### The Problem with Rotations

Rotations live on $SO(2)$ (2D) or $SO(3)$ (3D) — these are manifolds, not vector spaces.

You can't just add: $R_{\text{new}} = R + \Delta R$ (the result isn't a rotation matrix).

### Solution: Retraction/Exp Map

Update on the manifold using the Lie algebra:

$$x_{\text{new}} = x \oplus \Delta x = x \cdot \exp(\Delta x)$$

where:
- $\Delta x$ is in the tangent space (Lie algebra) — a regular vector
- $\exp$ maps back to the manifold

### For SE(2) — What OKS Uses

A pose is $(x, y, \theta)$. The "manifold" aspect is $\theta$ wrapping around.

For small problems: just use Euler angles and handle wrapping.
For large SLAM: use proper $SE(2)$ Lie group operations.

### In Code (Ceres Manifold)

```cpp
// Tell Ceres that this parameter block lives on SO(3)
problem.SetManifold(rotation_params, new ceres::EigenQuaternionManifold);

// For SE(3) — compose rotation + translation manifolds
auto se3_manifold = new ceres::ProductManifold<
    ceres::EigenQuaternionManifold,    // rotation (4 params, 3 dof)
    ceres::EuclideanManifold<3>         // translation (3 params, 3 dof)
>;
```

---

## 5. Solvers: g2o, GTSAM, Ceres

### g2o (General Graph Optimization)

Originally for SLAM, the most widely used graph optimizer in robotics.

```cpp
// g2o setup
g2o::SparseOptimizer optimizer;
auto solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(
        g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>>()
    )
);
optimizer.setAlgorithm(solver);

// Add vertices (variables)
auto v = new g2o::VertexSE2();
v->setId(0);
v->setEstimate(g2o::SE2(0, 0, 0));
optimizer.addVertex(v);

// Add edges (factors)
auto e = new g2o::EdgeSE2();
e->setVertex(0, optimizer.vertex(0));
e->setVertex(1, optimizer.vertex(1));
e->setMeasurement(g2o::SE2(dx, dy, dtheta));
e->setInformation(Omega);  // 3x3 information matrix
optimizer.addEdge(e);

// Optimize
optimizer.initializeOptimization();
optimizer.optimize(20);  // 20 iterations
```

### GTSAM (Georgia Tech Smoothing and Mapping)

More modern API, supports incremental solving (iSAM2).

```cpp
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

gtsam::NonlinearFactorGraph graph;
gtsam::Values initial;

// Prior on first pose
auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.05));
graph.addPrior(0, gtsam::Pose2(0, 0, 0), prior_noise);

// Odometry factor
auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));
graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2>>(0, 1, gtsam::Pose2(1, 0, 0), odom_noise);

// Initial estimates
initial.insert(0, gtsam::Pose2(0, 0, 0));
initial.insert(1, gtsam::Pose2(1, 0, 0));

// Optimize
gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial);
gtsam::Values result = optimizer.optimize();
```

### Ceres for Graph Optimization

Ceres isn't graph-specific but can solve any NLLS, including pose graphs:

```cpp
for (const auto& edge : pose_graph_edges) {
    ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<
        PoseGraphError, 3, 3, 3>(  // 3 residuals, 3+3 params (SE2)
            new PoseGraphError(edge.measurement, edge.information));
    problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                            poses[edge.from].data(),
                            poses[edge.to].data());
}
```

### Comparison

| Feature | g2o | GTSAM | Ceres |
|---------|-----|-------|-------|
| Purpose | Graph optimization | Factor graph inference | General NLLS |
| Incremental solve | Manual | iSAM2 (built-in) | No (batch only) |
| Manifolds | Built-in for SE2/SE3 | Built-in, extensive | Manual via `SetManifold` |
| Marginals / covariance | Basic | Full Bayes tree | Via `Covariance` class |
| Sparsity exploitation | Automatic | Automatic | Manual solver selection |
| Learning curve | Medium | Higher | Lower (general API) |
| Best for | Classic SLAM | Modern SLAM, multi-sensor | Calibration, small graph problems |

---

## 6. Robust Graph Optimization

### The Outlier Problem

Wrong loop closures (perceptual aliasing) create edges with completely wrong measurements. Standard least-squares faithfully incorporates them → corrupted map.

### Approaches

| Method | How | Ceres / g2o support |
|--------|-----|---------------------|
| Robust kernels | HuberLoss, CauchyLoss on each edge | Both |
| Switchable constraints | Add binary switch variable per edge | g2o (with plugin) |
| Max-mixtures | Model each edge as mixture of inlier/outlier | GTSAM (built-in) |
| DCS (Dynamic Covariance Scaling) | Scale information matrix based on residual | g2o, manual in Ceres |
| Graduated Non-Convexity (GNC) | Start with convex, gradually increase robustness | GTSAM |

### GNC — The Modern Approach

```
1. Start with standard (convex) least squares
2. Solve
3. Increase robustness parameter (make loss function more aggressive)
4. Solve again, warm-started
5. Repeat until convergence with the target robust loss
```

This avoids the problem of robust estimators having many local minima — the graduated approach tracks the global minimum as robustness increases.

---

## 7. Incremental Optimization (iSAM2)

### The Problem with Batch Optimization

Re-optimizing the entire graph after every new measurement is $O(n^3)$ — too slow for real-time.

### iSAM2 Solution

Maintain a **Bayes tree** — a data structure that:
- Encodes the factored information matrix
- Allows **incremental updates** — only re-solve the affected part
- Provides marginal covariances cheaply

When a new factor arrives:
1. Add it to the Bayes tree
2. Identify which variables are affected (the "clique" in the tree)
3. Re-eliminate only those variables
4. Cost: $O($ affected variables $)$ instead of $O(n^3)$

```cpp
// GTSAM incremental
gtsam::ISAM2 isam2;

// At each timestep:
gtsam::NonlinearFactorGraph new_factors;
gtsam::Values new_values;
new_factors.emplace_shared<BetweenFactor<Pose2>>(...);
new_values.insert(current_id, initial_estimate);

isam2.update(new_factors, new_values);
gtsam::Values current_estimate = isam2.calculateEstimate();
```

---

## 8. OKS Connections

### Navigation Estimator

The OKS navigation estimator uses an EKF, not full graph optimization. But the concepts connect:

| Graph concept | EKF equivalent |
|---------------|---------------|
| Prior factor | State prediction ($\hat{x}_{k|k-1}$) |
| Measurement factor | Measurement update (Kalman gain) |
| Information matrix $\Omega$ | Inverse covariance $P^{-1}$ |
| Optimization → MAP estimate | EKF update → conditional mean |
| Marginal covariance | Kalman covariance $P_{k|k}$ |

**Why the connection matters for RCA:**
- Understanding the information matrix helps debug why certain sensor failures cause estimator divergence
- The "graph view" explains why losing sensorbar measurements degrades pose accuracy — you're removing high-information factors
- Loop closure ↔ re-localization: when the robot re-detects a known landmark

### When Would OKS Need Full Graph Optimization?

- Multi-robot map merging
- Offline map refinement
- Full SLAM (if the current landmark-based approach is replaced)
- Calibration of multi-sensor rigs

---

## Summary: What to Carry Forward

| Concept | Where it's used next |
|---------|---------------------|
| Sparse Hessian structure | 07 (sparse linear algebra, fill-in) |
| Manifold updates (Lie groups) | Any SE(2)/SE(3) optimization |
| Factor graph formulation | RCA (understanding estimator as factor graph) |
| Robust optimization | Any SLAM or calibration with outliers |
| g2o, GTSAM, Ceres comparison | Choosing the right tool for the problem |
| iSAM2 incremental solving | Real-time SLAM understanding |
