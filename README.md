
# Tutorial: Coherent Point Drift (CPD) Algorithm for Point Set Registration  
### Comprehensive Breath Examination: Parth Ganeriwala  
In this tutorial, we will explore the Coherent Point Drift (CPD) algorithm for point set registration. Point set registration is the process of aligning two or more point clouds to find their spatial transformation. This is a fundamental problem in computer vision, robotics, medical imaging, and many other fields. CPD is a popular method for solving this problem due to its robustness and efficiency.  
  
## Background  
  
Before we dive into the details of CPD, let's briefly discuss the problem of point set registration and the Iterative Closest Point (ICP) algorithm, which is one of the most widely used methods for point set registration.  
  
### Point Set Registration  
  
Given two sets of points \($X = {x_1, x_2, ..., x_N}$\) and \($Y = {y_1, y_2, ..., y_M}$\), the goal of point set registration is to find a transformation \(T\) such that when applied to \(X\), the transformed points are aligned as closely as possible to \(Y\).

### Iterative Closest Point (ICP) Algorithm  
  
ICP is an iterative optimization algorithm commonly used for point set registration. It alternates between two steps:  
  
1. **Correspondence Estimation**: Given the current transformation \(T\), find the correspondence between points in \(X\) and \(Y\).  
2. **Transformation Estimation**: Given the correspondence, estimate the transformation \(T\) that aligns \(X\) with \(Y\).  
  
ICP repeats these steps until convergence.  
  
While ICP is effective, it can be sensitive to noise and outliers, and it may converge to local minima. CPD, on the other hand, addresses some of these limitations by incorporating coherence information and introducing soft correspondences.

## Coherent Point Drift (CPD) Algorithm  
  
The Coherent Point Drift (CPD) algorithm, introduced by Andriy Myronenko and Xubo Song in their paper "Point Set Registration: Coherent Point Drift" (2009), is a powerful method for point set registration. CPD extends traditional point set registration methods by introducing a probabilistic framework that models the point set correspondences as a Gaussian Mixture Model (GMM). This allows CPD to handle noise, outliers, and partial overlap between point sets more effectively than traditional methods like ICP.  
  
### Algorithm Overview  
  
The CPD algorithm can be summarized in the following steps:  
  
1. **Initialization**: Initialize the transformation \(T\) and the parameters of the Gaussian Mixture Model (GMM).  
2. **Expectation Step**: Compute soft correspondences between points in \(X\) and \(Y\) based on the current transformation and the GMM.  
3. **Maximization Step**: Update the parameters of the GMM and estimate the transformation \(T\) that maximizes the likelihood of the correspondences.  
4. **Iteration**: Repeat the Expectation and Maximization steps until convergence.  
  
Now, let's go through each step of the CPD algorithm in more detail.  
  
### Expectation Step  
  
In the Expectation step, we compute soft correspondences between points in \(X\) and \(Y\). This is done by estimating the posterior probability that a point in \(X\) corresponds to a point in \(Y\) given the current transformation \(T\) and the parameters of the GMM. The soft correspondences are computed using Bayes' theorem:  
  
$$
P(w_{ij} | x_i, y_j, T) = \frac{\pi_j \mathcal{N}(y_j | \mu_j, \Sigma_j)}{\sum_{k=1}^M \pi_k \mathcal{N}(y_j | \mu_k, \Sigma_k)}  
$$
  
where:  
- $(w_{ij})$ is the soft correspondence weight between $(x_i)$ and $(y_j)$.  
- $(\pi_j)$ is the mixing coefficient of the $(j^{th})$ component of the GMM.  
- $(\mathcal{N}(y_j | \mu_j, \Sigma_j))$ is the Gaussian distribution representing the $(j^{th})$ component of the GMM.  
- The denominator is the normalization term ensuring that the weights sum up to 1.  
  
### Maximization Step  
  
In the Maximization step, we update the parameters of the GMM and estimate the transformation \(T\) that maximizes the likelihood of the soft correspondences. This involves solving an optimization problem to find the optimal transformation and updating the parameters of the GMM using the weighted point correspondences.

## Point Cloud Registration with CPD Example Code 
#### (https://siavashk.github.io/2017/05/14/coherent-point-drift/)
Let's start off with a simple toy example. Assume that we have two point clouds $X = $\{$ X1, X2, X3 $\} and $Y = $\{$ Y1, Y2, Y3 $\} . These point clouds are shown in Figure 1 with red and blue circles, respectively. Our goal is to find the transformation that best aligns the two point clouds.

In this toy example, the unknown transformation is a rotation around the origin (parameterized by $\theta$ followed by a translation (parameterized by \(t\)). Assume, the actual value of the unknown parameters is \{$ \theta=30^\circ, t=(0.2, 0.2) $\}. We can use numpy to define the two point clouds as seen in the following code snippet:

 ```python
import numpy as np

# transformation parameters
theta = np.pi/6.0
t = np.array([[0.2], [0.2]])

# rotation matrix
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
X = np.array([[0, 0, 10], [0, 10, 0]])
Y = np.dot(R, X) + t

xLabels = ["X1", "X2", "X3"]
yLabels = ["Y1", "Y2", "Y3"]
```

Plotting the two point clouds results in Figure 1. Now, since this is a toy example, we already know the correspondences between points in the two point clouds. The corresponding points are linked using the black dashed line. If the correspondences are known, the solution to the rigid registration is known as the orthogonal Procrustes problem:

$$\mathrm{argmin}_{R,t}\Vert{X - RY - t}\Vert^2, \quad \mathrm{s.t} \quad R^TR=I$$

![Point Cloud Registration](/assets/cpd/registration1_1_0.png)<br/>

## Missing Correspondences
When correspondence is not explicitly known, point cloud registration algorithms implicitly assume that correspondence can be inferred through point proximity. In other words, points that are spatially close to each other correspond to one another.

We can assign an arbitrary correspondence probability to point clouds based on proximity. Figure 2 shows an example probability distribution based on proximity.

Points that are closer than a radius of \(r=0.2\) would confident matches, and we would assign a correspondence confidence of \(p=1.0\) to them. Pairs such as \(\(X1, Y1\)\) and \(\(X2, Y2\)\) pairs have a distance between \(r=0.2\) and \(r=0.4\) units are probable but not confident matches, so we could assign a probability of \(p=0.5\) to them. Beyond this, there is probably no correspondence, so our probability would drop to zero.

Even though this approach is quite simple, it provides two distinct advantages. First, it allows us to assign correspondences so that we can solve the registration as a Procrustes problem. Furthermore, it also allows us to weigh the loss functional according to the correspondence probability.
<br>
![Point Cloud Correspondences](/assets/cpd/registration1_2_0.png)<br/>

## Gaussian Mixture Models
We will now side step from the point cloud registration problem briefly. Instead of dealing with \(X, Y\) point clouds directly, we construct a GMM from the moving point cloud, \(Y\), and treat \(X\) as observations from that GMM. In Figure 3, we have constructed a GMM where the three Gaussians have a variance of 0.75 units. Blue points, i.e. Gaussian centroids, are the transformed moving points (\(Y\)). Red points, i.e. the fixed point cloud \(X\), are observations from this GMM. Isocontours represent the log-likelihood that red points are sampled from this GMM.

<br>![Constructed GMM](/assets/cpd/registration1_3_0.png)<br/>

## GMM-based Registration
In order to perform registration, we have to solve correspondence and moving point cloud transformation problems simultaneously. This is done through expectation-maximization (EM) optimization. To solve the correspondence problem, we need to find which Gaussian the observed point cloud was sampled from (E-step). This provides us with correspondence probability, similar to Figure 2. Once correspondences probabilities are known, we maximize the negative log-likelihood that the observed points were sampled from the GMM with respect to transformation parameters (M-step).

## Expectation Step
In Figure 3, if there was only one Gaussian component in the mixture, then the probability that a point \(x\) is sampled from this Gaussian is given using probability density distribution of the [multivairate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function). For the 2D case, with isotropic Gaussians, this simplifies to:

$$p(X) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp({-\frac{\Vert{X - RY - t}\Vert^2}{2\sigma^2}})$$

However, since we are dealing with multiple Gaussians, we need to normalize this probability by the contribution of all Gaussian centroids. In the pycpd package, this is achieved (minor tweaks to simplify the explanation) using the following snippet:

```python
import numpy as np

def EStep(X, Y, sigma2):
  M = Y.shape[0] # number of moving points
  D = Y.shape[1] # dimensionality of moving points
  N = X.shape[0] # number of fixed points
  # Probability matrix: p_{ij} is the probability
  # that moving point i corresponds to fixed point j
  P = np.zeros((M, N))

  for i in range(0, M):
      diff     = X - np.tile(Y[i, :], (N, 1))
      diff    = np.multiply(diff, diff)
      P[i, :] = P[i, :] + np.sum(diff, axis=1)

  P = np.exp(-P / (2 * sigma2))
  den = np.sum(P, axis=0)
  den = np.tile(den, (M, 1))
  den[den==0] = np.finfo(float).eps

  P = np.divide(P, den)
  Pt1 = np.sum(P, axis=0)
  P1  = np.sum(P, axis=1)
  Np  = np.sum(P1)
  return P, Pt1, P1, Np
```

## Maximization Step
Once correspondence probabilities are known, i.e. \(P\), we can solve for the transformation parameters. In the case of rigid registration, these transform parameters are the rotation matrix and the translation vector. In the pycpd package, this is achieved using the following snippet:

```python
import numpy as np

def MStep(X, Y, P):
  s, R, t, A, XX = updateTransform(X, Y, P, P1, Np)
  sigma2 = updateVariance(R, A, XX, Np, D)

def updateTransform(X, Y, P):
  muX = np.divide(np.sum(np.dot(P, X), axis=0), Np)
  muY = np.divide(np.sum(np.dot(np.transpose(P), Y), axis=0), Np)

  XX = X - np.tile(muX, (N, 1))
  YY = Y - np.tile(muY, (M, 1))

  A = np.dot(np.transpose(XX), np.transpose(P))
  A = np.dot(A, YY)

  U, _, V = np.linalg.svd(A, full_matrices=True)
  C = np.ones((D, ))
  C[D-1] = np.linalg.det(np.dot(U, V))

  R = np.dot(np.dot(U, np.diag(C)), V)

  YPY = np.dot(np.transpose(P1), np.sum(np.multiply(YY, YY), axis=1))

  s = np.trace(np.dot(np.transpose(A), R)) / YPY

  t = np.transpose(muX) - s * np.dot(R, np.transpose(muY))
  return s, R, t, A, XX

def updateVariance(R, A, XX, Np, D):
  trAR = np.trace(np.dot(A, np.transpose(R)))
  xPx = np.dot(np.transpose(Pt1), np.sum(np.multiply(XX, XX), axis =1))
  sigma2 = (xPx - s * trAR) / (Np * D)
  return sigma2
```

## Conclusion  
  
In this tutorial, we provided an overview of the Coherent Point Drift (CPD) algorithm for point set registration. We discussed the CPD algorithm's steps, Expectation and Maximization, and compared it to the Iterative Closest Point (ICP) algorithm. We also provided example code demonstrating how to implement CPD for point set registration using synthetic data.  
  
CPD is a powerful method for point set registration, capable of handling noise, outliers, and partial overlap between point clouds. However, like any algorithm, CPD has its limitations and may require parameter tuning for optimal performance in specific applications. Experimentation and validation on real-world data are essential for assessing the effectiveness of CPD in practical scenarios.