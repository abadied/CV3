import numpy as np
import matplotlib.pyplot as plt


def sampler(num_iterations, temp):
    drawn_lttice = np.random.choice([-1,1], (8,8))
    lattice = np.zeros((10,10))
    lattice[1:9, 1:9] = drawn_lttice
    # Temp = {1,1.5,2}

    for i in range(num_iterations):
        for x in range(1,9):
            for y in range(1,9):
                lattice[x][y] = sample(lattice,x,y,temp)
    return lattice

# Gibbs Sampling from the IsingModel Prior
def sample(lattice, x, y, temp):
    # get values of neighbors vertices
    pos_prior = np.exp((1/temp)*lattice[x,y]*(lattice[x - 1, y]+
                                                lattice[x + 1, y]+
                                                lattice[x, y - 1]+
                                                lattice[x, y + 1]))

    neg_prior = np.exp((1/temp)*-1*lattice[x,y]*(lattice[x - 1, y]+
                                                lattice[x + 1, y]+
                                                lattice[x, y - 1]+
                                                lattice[x, y + 1]))

    # Normalize so the probabilities will sum to 1
    Z = pos_prior+neg_prior

    return np.random.choice([-1,1], p=[neg_prior/Z, pos_prior/Z])


def main():
    lattice = sampler(100, 1.5)
    plt.imshow(lattice, cmap="Greys", interpolation="None")
    plt.show()
if __name__ == "__main__":
    main()