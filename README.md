# TIF345-Advanced-simulation-and-machine-learning

Repository for projects in the Chalmers course "TIF345 / FYM345 Advanced simulation and machine learning" 2020. 

Authors: Sebastian Holmin and Erik Andersson

The code in this repository was used to produce the following four reports:

## [Cosmological models](TIF345_Project_1__Cosmological_models.pdf)

In this project, we use utilize the Supernova Cosmology Project (SCP) data to analyze and compare cosmological models. The SCP 2.1 dataset contains detailed measurements of  theredshift, z, and distance moduli μ, of several supernovea, which we utilize to perform Bayesianparameter estimations and model comparisons.

## [Alloy cluster expansions](TIF345_Project_2a__Alloy_cluster_expansions.pdf)

In this report we investigate the issue of parameter selection and estimation in cluster expansions of alloys. To do this we use the <code>icet</code> package which implements symmetry transformations to expand the mixing energy of alloy structures into 

![image](https://user-images.githubusercontent.com/16863941/150383786-f6a7e32d-60f4-4a2d-9de9-c226ecfc94b5.png)

where ![formula](https://render.githubusercontent.com/render/math?math=N_\alpha) is the number of a ![formula](https://render.githubusercontent.com/render/math?math=\alpha)-clusters per atom and ![formula](https://render.githubusercontent.com/render/math?math=J_\alpha) is the effective cluster interaction (ECI), which are the parameters that we seek to estimate from energy data.

Assuming i.i.d. errors this can be written in matrix notation as ![formula](https://render.githubusercontent.com/render/math?math=\boldsymbol{E}=\boldsymbol{X}\boldsymbol{J}%2B\epsilon), with ![formula](https://render.githubusercontent.com/render/math?math=\epsilon\sim\mathcal{N}(0,\sigma^2)) and thus the likelihood function is given by

![image](https://user-images.githubusercontent.com/16863941/150384001-3f6a15b0-d7a7-40c0-bbf9-abfe8f86a364.png)

The Bayesian and Akaike information criteria are defined at the maximum likelihood, which can be shown to be equivalent to 

![image](https://user-images.githubusercontent.com/16863941/150384093-80b7c224-9e17-4b15-b71e-8cfb2c3846ad.png)

where MSE is the mean squared error.

## [Bayesian Optimization: Searching for the global minima](TIF345_Project_2b__Bayesian_Optimization__Searching_for_the_global_minima.pdf)

In this project we investigate the use of Gaussian Processes (GP) to model the potential energy surface (PES) for adding a Au atom to a Au slab, i.e. the difference in average energy per atom between the slab with and without the extra atom. To sample the energy we use an embedded medium theory (EMT) calculator provided in the <code>asap3</code> package. Sampling the energy this way is resource intensive, so GPs are likely well suited method for reducing the computational time for modeling the PES.

## [A Galton board on a rocking ship](TIF345_Project_3__A_Galton_board_on_a_rocking_ship.pdf)

In this report we investigate the use of the approximate Bayesian computation (ABC) algorithm, supported by a neural network (NN), to reverse engineer the parameters for a toy model of a Galton board on a rocking ship.

A Galton board (bean machine) is a device that produces a normal distribution by utilizing the law of big numbers. It consists of rows of pegs where balls can roll a step to the left or the right at each row. Our toy model has 31 rows, giving 32 possible end positions for each ball, and two parameters ![formula](https://render.githubusercontent.com/render/math?math=\alpha) and ![formula](https://render.githubusercontent.com/render/math?math=s). The parameter ![formula](https://render.githubusercontent.com/render/math?math=\alpha) describes a simplified moment of inertia, i.e. the tendency for a ball to continue rolling in the same direction again for the next peg, while ![formula](https://render.githubusercontent.com/render/math?math=s) describes the incline of the rocking ship that the Galton board is situated on. The probability of a ball rolling to the right is then given by 

![image](https://user-images.githubusercontent.com/16863941/150390102-699d04c0-137b-4c4a-9130-d3842f1491ca.png)

where ![formula](https://render.githubusercontent.com/render/math?math=\alpha\in[0,0.5]) and ![formula](https://render.githubusercontent.com/render/math?math=s\in[-0.25,0.25]) and ![formula](https://render.githubusercontent.com/render/math?math=M=-0.5) if the ball previously rolled to the left and ![formula](https://render.githubusercontent.com/render/math?math=0.5) if it rolled to the right.

We are faced with the task of determining an unknown (but constant) ![formula](https://render.githubusercontent.com/render/math?math=\alpha) from a 'black box' function that simulates the final positions of 1000 balls. This is done for an unknown, randomly chosen, latent variable ![formula](https://render.githubusercontent.com/render/math?math=s). To help us with this task we will implement our own simulator where we can control the parameters and analyse the behaviour. 

Phrased in a Bayesian language, given a (set of) simulated distributions ![formula](https://render.githubusercontent.com/render/math?math=y_m), find the posterior distribution 

![image](https://user-images.githubusercontent.com/16863941/150390332-f25744d9-86f2-45a5-a2f1-31d8885f1f34.png)

where we have first used the law of total probability to write the likelihood as marginalized over the latent variable ![formula](https://render.githubusercontent.com/render/math?math=s) and then Bayes theorem to write the posterior in terms of the likelihood. As this is a toy model, we will assume that ![formula](https://render.githubusercontent.com/render/math?math=s) is drawn uniformly in its allowed interval for every run of the 'black box' simulation, and that the 'true' ![formula](https://render.githubusercontent.com/render/math?math=\alpha) was chosen randomly from uniform distribution. That is, we will assume that the priors ![formula](https://render.githubusercontent.com/render/math?math=\pi(\alpha)) and ![formula](https://render.githubusercontent.com/render/math?math=\pi(s)) are uniform.

The accuracy of the posterior can be increased using the results from several experiment outcomes. Given a set ![formula](https://render.githubusercontent.com/render/math?math=y_m^{i}), for ![formula](https://render.githubusercontent.com/render/math?math=i=1,...,N), the total posterior is given by 

![image](https://user-images.githubusercontent.com/16863941/150390491-d8d282c2-e455-49ac-a966-773e4bd2e6d9.png)

where we have use the fact that ![formula](https://render.githubusercontent.com/render/math?math=\pi(\alpha)) is uniform to insert the posterior.
