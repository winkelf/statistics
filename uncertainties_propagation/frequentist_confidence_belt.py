import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_gaussian_numbers(N, mean, std_dev):
    numbers = np.random.normal(loc=mean, scale=std_dev, size=N)
    return numbers

npoints = 100

def toy():

    means     = []
    quantiles = []
    quantiles_u_1sig = []
    quantiles_d_1sig = []
    quantiles_u_2sig = []
    quantiles_d_2sig = []
    quantiles_u_3sig = []
    quantiles_d_3sig = []  

    for i in tqdm(np.linspace(0,2,npoints+1), desc=f'Generating confidence belt with {npoints} points'):
        #print(f"mu = {i}")
        N       = 10000  
        mean    = i  
        std_dev = 1 
    
        gaussian_numbers = generate_gaussian_numbers(N, mean, std_dev)
    
        statistic_distribution = []
    
        for j in range(5000):
            sampled_numbers = np.random.choice(gaussian_numbers, size=100, replace=False)
            average = sum(sampled_numbers)/len(sampled_numbers)
            statistic_distribution.append(average)
    
        q = np.quantile(statistic_distribution, 0.95)
       
        cl_1sig = 0.6827
        cl_2sig = 0.9545
        cl_3sig = 0.9973

        alpha1 = 1-cl_1sig
        alpha2 = 1-cl_2sig
        alpha3 = 1-cl_3sig
        
        qd_1sig = np.quantile(statistic_distribution, 1-alpha1/2) #0.975)
        qu_1sig = np.quantile(statistic_distribution, alpha1/2  ) #0.025 )
        qd_2sig = np.quantile(statistic_distribution, 1-alpha2/2) 
        qu_2sig = np.quantile(statistic_distribution, alpha2/2  ) 
        qd_3sig = np.quantile(statistic_distribution, 1-alpha3/2) 
        qu_3sig = np.quantile(statistic_distribution, alpha3/2  ) 

        #print(q)
        means.append(i)
        quantiles.append(q)
        quantiles_u_1sig.append(qu_1sig)
        quantiles_d_1sig.append(qd_1sig)
        quantiles_u_2sig.append(qu_2sig) 
        quantiles_d_2sig.append(qd_2sig)
        quantiles_u_3sig.append(qu_3sig) 
        quantiles_d_3sig.append(qd_3sig)

        if (i==1):
            # Plotting the generated numbers
            plt.hist(statistic_distribution, bins=50, density=True, alpha=0.7, color='blue')
            plt.axvline(q, color='red', linestyle='dashed', linewidth=2, label='5% quantile')
            plt.title(f"Statistic distribution")
            plt.xlim([-4,14])
            plt.xlabel("Value")
            plt.ylabel("Probability Density")
            plt.grid(True)
            plt.savefig(f'statistic_{i}.pdf')
            plt.close()

    ## Measurements/Calculate coverage
    #
    N_observations = 1000000

    counts_1sig = 0
    counts_2sig = 0
    counts_3sig = 0

    #for t in range(N_observations):
    for t in tqdm(range(N_observations), desc=f'Simulating measurements'):
        new_mean = np.random.uniform(0,2)
        #print(f"Truth mean: {new_mean}")
        gaussian_numbers = generate_gaussian_numbers(1000, new_mean, 1)
        sampled_numbers = np.random.choice(gaussian_numbers, size=100, replace=False)
        average = sum(sampled_numbers)/len(sampled_numbers)
    
        N = average
    
        closest_u_1sig = min(quantiles_u_1sig, key=lambda x: abs(x - N))
        closest_d_1sig = min(quantiles_d_1sig, key=lambda x: abs(x - N))
        index_u_1sig   = min(range(len(quantiles_u_1sig)), key=lambda i: abs(quantiles_u_1sig[i] - N))
        index_d_1sig   = min(range(len(quantiles_d_1sig)), key=lambda i: abs(quantiles_d_1sig[i] - N))

        closest_u_2sig = min(quantiles_u_2sig, key=lambda x: abs(x - N))
        closest_d_2sig = min(quantiles_d_2sig, key=lambda x: abs(x - N))
        index_u_2sig   = min(range(len(quantiles_u_2sig)), key=lambda i: abs(quantiles_u_2sig[i] - N))
        index_d_2sig   = min(range(len(quantiles_d_2sig)), key=lambda i: abs(quantiles_d_2sig[i] - N))

        closest_u_3sig = min(quantiles_u_3sig, key=lambda x: abs(x - N))
        closest_d_3sig = min(quantiles_d_3sig, key=lambda x: abs(x - N))
        index_u_3sig   = min(range(len(quantiles_u_3sig)), key=lambda i: abs(quantiles_u_3sig[i] - N))
        index_d_3sig   = min(range(len(quantiles_d_3sig)), key=lambda i: abs(quantiles_d_3sig[i] - N))

        if t==0:
            #plt.plot(quantiles, means)
            #plt.plot(quantiles_u_1sig, means, color='cornflowerblue')
            #plt.plot(quantiles_d_1sig, means, color='cornflowerblue')
            plt.fill_betweenx(means, quantiles_d_1sig, quantiles_u_1sig, color='cornflowerblue', alpha=0.9, label='1$\sigma$ CL')
            plt.fill_betweenx(means, quantiles_d_2sig, quantiles_u_2sig, color='crimson',        alpha=0.2, label='2$\sigma$ CL')
            plt.fill_betweenx(means, quantiles_d_3sig, quantiles_u_3sig, color='darkgreen',      alpha=0.1, label='3$\sigma$ CL')
            plt.axvline(average, color='red', linestyle='dashed', linewidth=0.5)#, label='obs')
            plt.text(average+0.05, 1, '$\hat{\mu}_{obs}$', color='red')
            plt.axhline(means[index_u_1sig], color='green', linestyle='dashed', linewidth=0.5)#, label=f'Upper lim: {means[index_u_1sig]}')
            plt.axhline(means[index_d_1sig], color='green', linestyle='dashed', linewidth=0.5)#, label=f'Lower lim: {means[index_d_1sig]}')
            plt.title(f"Neyman confidence belt")
            plt.xlabel("$\hat{\mu}$")
            plt.ylabel("$\mu$")
            #plt.yscale('log')
            #plt.xscale('log')
            plt.legend()
            #plt.grid(True)
            plt.savefig("belt.pdf")
    
        ulim_1sig = means[index_u_1sig] 
        dlim_1sig = means[index_d_1sig]
   
        ulim_2sig = means[index_u_2sig] 
        dlim_2sig = means[index_d_2sig]

        ulim_3sig = means[index_u_3sig] 
        dlim_3sig = means[index_d_3sig]

        inside_1sig = new_mean<ulim_1sig and new_mean>dlim_1sig
        inside_2sig = new_mean<ulim_2sig and new_mean>dlim_2sig
        inside_3sig = new_mean<ulim_3sig and new_mean>dlim_3sig

        if (inside_1sig): counts_1sig+=1
        if (inside_2sig): counts_2sig+=1
        if (inside_3sig): counts_3sig+=1

    return counts_1sig/N_observations, counts_2sig/N_observations, counts_3sig/N_observations

if __name__ == "__main__":

    out = toy()

    print(f"Coverage 1 sigma: {out[0]}")
    print(f"Coverage 2 sigma: {out[1]}")
    print(f"Coverage 3 sigma: {out[2]}")

