import numpy as np
import matplotlib.pyplot as plt


def sample_lattice(width):
    # initiate with zeros
    lattice = np.zeros((width + 2, width + 2))
    # samples fairly to the inner lattice without 0 bounds
    lattice[1: width + 1, 1: width + 1] = np.random.choice([-1, 1], (width, width))
    return lattice


def sweep(lattice, width, temp, posterior=None, sigma=2, icm=False):
    for i in range(1, width + 1):
        for j in range(1, width + 1):
            sum_neighbors = lattice[i, j - 1] + lattice[i, j + 1] + \
                            lattice[i + 1, j] + lattice[i - 1, j]
            prob_plus = (1 / temp) * sum_neighbors
            prob_min = (-1 / temp) * sum_neighbors

            if posterior is not None:
                prob_plus -= ((posterior[i, j] ** 2) - 1) / (2 * (sigma ** 2))
                prob_min -= ((posterior[i, j] ** 2) + 1) / (2 * (sigma ** 2))

            probs = np.exp([prob_plus, prob_min])
            probs = probs / probs.sum()

            if icm:
                lattice[i, j] = 1 if prob_plus > prob_min else -1
            else:
                lattice[i, j] = np.random.choice([1, -1], p=probs)

    return lattice


def gibbs_methods_sampler(width, temp, num_of_iter, num_of_sweeps, ergodic=False, warm_up=100):

    expectation_x11_x22 = 0
    expectation_x11_x88 = 0

    lattice = sample_lattice(width)

    # warm up for the ergodic method
    if ergodic:
        for swe in range(warm_up):
            lattice = sweep(lattice, width, temp)

    for i in range(num_of_iter):

        for swe in range(num_of_sweeps):
            lattice = sweep(lattice, width, temp)

        # iteration update
        expectation_x11_x22 += lattice[1, 1] * lattice[2, 2]
        expectation_x11_x88 += lattice[1, 1] * lattice[8, 8]

        if not ergodic:
            lattice = sample_lattice(width)

    expectation_x11_x22 /= num_of_iter
    expectation_x11_x88 /= num_of_iter

    if ergodic:
        print('Ergodicity Method')
    else:
        print('Independent Method')

    print('Temp: ' + str(temp))
    print('E x11 * x22: ' + str(expectation_x11_x22))
    print('E x11 * x88: ' + str(expectation_x11_x88))

    return


def gibbs_sampler(width, temp, num_sweeps, posterior=None, icm=False):
    lattice = sample_lattice(width)

    for _ in range(num_sweeps):
        lattice = sweep(lattice=lattice, width=width, temp=temp, posterior=posterior, icm=icm)

    return lattice


def main():
    # Part 1
    temps = [1., 1.5, 2.]
    width = 8
    for temp in temps:
        gibbs_methods_sampler(width, temp, 10000, 25)
        gibbs_methods_sampler(width, temp, 24900, 1, ergodic=True)

    # add plot
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))

    # Part 2
    width = 100
    for idx, temp in enumerate(temps):
        # 50 rast sweeps
        init_lattice = gibbs_sampler(width=width, temp=temp, num_sweeps=50)
        # plot init lattice
        axes[idx][0].imshow(init_lattice[1: width + 1, 1: width + 1], cmap="Greys", interpolation="None")

        # add noise to lattice
        noise_lattice = init_lattice.copy()
        noise_lattice[1: width + 1, 1: width + 1] += 2 * np.random.standard_normal(size=(width, width))
        # plot noisy lattice
        axes[idx][1].imshow(noise_lattice[1: width + 1, 1: width + 1], cmap="Greys", interpolation="None")

        # denoise
        denoised_lattice = gibbs_sampler(width=width, temp=temp, num_sweeps=50, posterior=noise_lattice)
        # plot denoised lattice
        axes[idx][2].imshow(denoised_lattice[1: width + 1, 1: width + 1], cmap="Greys", interpolation="None")
        # plot icm lattice
        icm_denoised_lattice = gibbs_sampler(width=width, temp=temp, num_sweeps=50, posterior=noise_lattice, icm=True)
        axes[idx][3].imshow(icm_denoised_lattice[1: width + 1, 1: width + 1], cmap="Greys", interpolation="None")

    axes[0][0].annotate("Initial Sample", xy=(0.5, 1), xytext=(0, 5),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
    axes[0][1].annotate("Noisy Image", xy=(0.5, 1), xytext=(0, 5),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
    axes[0][2].annotate("Sample Posterior", xy=(0.5, 1), xytext=(0, 5),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
    axes[0][3].annotate("Sample ICM", xy=(0.5, 1), xytext=(0, 5),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
    plt.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == '__main__':
    main()
