# Ride sharing data challenge

---
## Contents

[The problem](#problem)

[Part 1 Approach](#1approach)

[Part 2 Approach](#2approach)

[Evaluating the approach](#exa)

[Summary and Future Directions](#future)


---





# <a name="problem">The problem</a>

Please note: I have made the description of the problem and solution purposely vague to make this notebook difficult to search for.

This repo has code that solves different aspects of vehicle routing problem variations. Namely, I'm interesed in understanding the potential benefits of clustering ride requests.

Part 1:
The first a script that solves vehicle scheduling for a small number of ride requests (e.g. for ride requests that are clustered together). I reformulated the problem as an optimization problem, and then implemented a script that builds the problem and finds a solution.

Part 2:
The second is an EDA of some easily searchable vehicle data, with some 'back of the envelope' calculations to approximate efficiency gains from clustering ride requests. 

These two components do not currently work together.
To merge the two components:
- Coordinates need to be standardized
- Distance metrics need to be standardized
- Some quick data wrangling should suffice to standardize the df format (no derived variables are needed)


# <a name="1approach">Part 1 Approach</a>

The script finds the min number of vehicles required to service a set of ride requests using a binary search over possible candidate solutions. 

Here's the script: [LINK!](https://github.com/iurrutia/legendary-bassoon/blob/master/local_routing/local_routing.py)


My script is an IP formulation (solved with PySCIPOpt) that looks for a feasible solution that satisfies all trip requests with k cars. I perform a binary search to find the smallest value k such that there exists a feasible solution. I assume that k ≤ p, where p is the number of clients, that pickups and dropoffs take zero time, and that all distances can be rounded to the nearest integer. The program creates itineraries (if feasible solutions are found), until it finds the smallest k for which a feasible solution exists, at which point it also prints out a message “Feasible itinerary achieved with k vehicles. No feasible solutions with fewer vehicles.” 

The constraints are explained in the script, which is heavily commented. The broad idea behind my approach was to create a graph with one vertex per dropoff location, pickup location, an imaginary source node that is free to travel out of, and expensive to travel into, and K imaginary sink nodes that are free to travel into, but expensive to travel out of.

Let N be the set of all nodes: pick-up and drop-off locations $N_{pu}$, $N_{do}$, a source node 0, and K sink nodes, $N_{sinks}$. Variable $x_{i,j}$ equals 1 if the edge (i,j) is used by a car in a tour, and equals zero otherwise. Variable $y_{i,j}$ equals 1 if car k visits node i and is zero otherwise. Variable wi is the time at which node i is departed in the final itinerary, and ci is the number of people in the car that visits node i when it departs node i, M
is ‘a very big number’ (functions as ‘+ infinity’), $E_i$ and $L_i$ are the upper and lower bounds of node i’s time window, $t_{i,j}$ is the time it takes to travel from i to j, and $d_i =1$ if i ∈ $N_{pu}$, $d_i =−1$ if i ∈ $N_{do}$, and $d_i =0$ otherwise. 


Note that this approach searches for feasibility and not optimality (thus there is no objective function in the linear program). The algorithm currently assumes Euclidean distances. 





# <a name="2approach">Part 2 Approach</a>

I proposed an algorithm which does the following:

- Group rides into time buckets
- Run k-means in each time bucket to obtain clusters of 'similar' ride requests within time buckets
- For each cluster, plan vehicle assignments and routes as follows:
    - Check whether the pickups or drop-offs are more *dispersed* 
    - Run TSP on whichever of the above is less *dispersed* (e.g. assume pick-ups are more dispersed, without loss of generality)
    - To assign passengers to vehicles, assign consecutive passengers in TSP solution to the same vehicle until the vehicle is full
    - Run a second TSP instance to determine drop-off order (without loss of generality)
    
    
### The following assumptions have been made:

- I assume drivers appear and disappear when required, and have not considered an optimal way of allocating drivers to the pooled rides identified in the algorithm. In other words, this approach ignores the problem of driver allocation by shift, thus also ignoring the scheduling problem that Via would have to solve to implement this approach.
- To calculate trip times, I have assumed that vehicles complete all trips at the average trip speed within each cluster. This ignores differences in speeds between highways/freeways and driving around slower parts of the city (e.g. driving through the middle of downtown).
- I have assumed that passengers are willing to wait over five minutes to be picked up by partitioning rides into 5-minute buckets, and then allowing for time to be picked up on top of those 5 minutes.
- The step of the algorithm that assigns passengers to vehicles ignores cases where a group of more than one passengers wish to ride together, and the current vehicle being assigned passengers cannot fit the whole group. Under the current algorithm, these parties would have to be spilt up instead of waiting for a taxi they could ride in together. This is probably not what passengers would choose to do, if given the option.
- I have approximated the time/distance to pick up and drop off passengers as <img src="https://render.githubusercontent.com/render/math?math=3d"> using the implicit assumption that the trip pickup and drop-off locations are uniformly distributed, which is not necessarily true. The approximation was found as follows:
    - Assuming <img src="https://render.githubusercontent.com/render/math?math=n"> locations are uniformly distributed, let <img src="https://render.githubusercontent.com/render/math?math=L_n^{*}">  be the solution to TSP on these <img src="https://render.githubusercontent.com/render/math?math=n"> locations. As <img src="https://render.githubusercontent.com/render/math?math=n \to \infty">, <img src="https://render.githubusercontent.com/render/math?math=L_n^{*}/\sqrt{n} \to \beta">, where the current upper bound on <img src="https://render.githubusercontent.com/render/math?math=\beta"> is a constant (<1). In our data set, the clusters tended to be quite large, so I assumed that the time required to pick up (or wlog drop off) passengers who are consecutive in a TSP solution within a cluster tends towards zero for larger values of *n*.
    - Without loss of generality, assume <img src="https://render.githubusercontent.com/render/math?math=d = d_{pu}">. Using a geometric argument, it can be shown that if we use euclidean distance (another assumption), I roughly approximate the distance required to drop passengers off by *3d* (The total efficiency from the algorithm is very sensitive to this approximation, and the approximation is not correctly bounded.)


# <a name="exa">Evaluating the approach</a>

The figure below shows the number of driver labour hours used through carpooling (new time worked) compared to the number of driver labour hours used to meet the same demand (old time worked) in the yellow taxi data set over the same time period. Note that the aggregation performs roughly the same on average per day of the week.


![](images/saved_day.jpg)


First, note that we can observe that the difference between using/not-using aggregation is higher on weekdays than on weekends, particularly between 5am and noon, which are times when demand on weekend falls below the average weekday demand. This suggests that aggregation can lead to more efficient use of driver labour on weekdays, during the day, particularly during peak hours. 

![](images/pickups_hr.jpg)

Second, note that the following figure also reveals that  aggregation only leads to modest gains in efficiency between midnight and 5am, and on Saturdays, compared to other days and times. From these observations, we can conclude that an aggregation algorithm should be optimized to be most effective during peak hours, on workdays, since a rudimentary approach shows these time windows offer the highest potential savings in driver labour costs.

![](images/day_hrSdiff.jpg)


# <a name="future">Summary and Future Directions</a>

Using the approximations detailed above, this algorithm meets ridership demand using 6,356,147 hours of driver labour, which is an overall reduction of 8,72,571 hours over the hours of labour used to satisfy ridership demand with un-pooled rides.

Computationally intensive steps are performed on small data sets. k-means runs in $O(n^2)$, and we are only ever running k-means on the trips within each time bucket, so these k-means can be run independently (in parallel) if needed. TSP will also need to be run only once within each cluster within each time bucket, so although TSP is NP-hard, it runs independently on relatively small point sets. 

Clear next steps to improve this project include:
- Implementing TSP, or an approximation of TSP to calculate a tighter bound on the true cost savings made possible by the proposed algorithm.
- Exploring solutions which include intermediate pickups and dropoffs.
