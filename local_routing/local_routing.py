"""# 0. SETUP """
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------

import sys
import pandas as pd
import math 
import datetime as dt
from scipy.spatial.distance import pdist, squareform
import numpy as np
from itertools import product
from pyscipopt import Model, quicksum


"""# 1. READ IN DATA, create dataframe and adjancency matrix between locations"""
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------

"""# 1.1 Make dataframe """
# --------------------------------------------------------------------

def wrangle(filename):

  """Function reads in file 'filename' of trip requests, and creates a dataframe of trip vertices, where
  each pickup and dropoff is considered as a separate node.
  The output dataframe has a first row (sink) and last row (source) of made-up nodes.
  Output dataframe has columns:
  x : x coordinate
  y : y coordinate
  E : earliest car can leave node
  L : latest a car can leave node
  Requester : name of passenger
  trip_n : trip id (trips are identified by the row number in original input)
  d : +1 if node is a pickup, -1 if node is a dropoff 
  """
  
  df = pd.read_csv(filename, sep = "\t", skiprows = 1, names = ('Requester', 'Trip','Depart After', 'Arrive Before', 'X1', 'Y1', 'X2', 'Y2'))
  
  # Separate pickups and dropoffs:
  pickups = df[['Requester','Depart After','Arrive Before', 'X1', 'Y1']].copy()
  pickups = pickups.rename(columns={"X1": "x", "Y1": "y"})
  pickups['d'] = 1
  
  dropoffs = df[['Requester','Depart After','Arrive Before', 'X2', 'Y2']].copy()
  dropoffs = dropoffs.rename(columns={"X2": "x", "Y2": "y"})
  dropoffs['d'] = -1
  
  # Concat pickup and dropoff dataframes:
  df2 = pickups.append(dropoffs)
  
  # Convert all times into datetime objects:
  df2['Depart After'] = pd.to_datetime(df2['Depart After'],format='%H:%M')
  df2['Arrive Before'] = pd.to_datetime(df2['Arrive Before'],format='%H:%M')
  
  # M_E/M_L = global min/max times, respectively.
  M_E = pd.to_datetime(df2['Depart After']).min()
  M_L = pd.to_datetime(df2['Arrive Before']).max()

  # Cleaning up the dataframe
  df2 = df2.reset_index()    
  df2 = df2.rename(columns={"Depart After": "E", "Arrive Before": "L", 'index':'trip_n'})
  
  # Adding artificial source/sink nodes (for the IP formulation)
  depot_s = [[0, 0, M_E, M_L,'source', -1, 0]]
  depot_t = [[0, 0, M_E, M_L,'sink', -1, 0]]
  dfs = pd.DataFrame(depot_s, columns=['x', 'y', 'E', 'L', 'Requester', 'trip_n', 'd'])
  dft = pd.DataFrame(depot_t, columns=['x', 'y', 'E', 'L', 'Requester', 'trip_n', 'd'])
  df_output = dfs.append(df2)
  df_ = df_output.append(dft)
  
  # Cleaning up after dataframe is assembled
  df_ = df_[['x', 'y', 'E','L','Requester','trip_n', 'd']]

  # Convert all times into minutes using helper function (for the IP formulation)
  df_['E'] =  df_['E'].map(mins_since_midnight)
  df_['L'] =  df_['L'].map(mins_since_midnight)
  
  df = df_.reset_index(drop=True)
  
  return df

"""# 1.2 Helper functions """
# --------------------------------------------------------------------

def distance(a, b):
  """returns the travel time (scaled euclidean distance) 
  between points a = [Xa, Ya], and b = [Xb, Yb]"""
  return (0.2)*math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def mins_since_midnight(time):
  """returns minutes since midnight"""
  zero_time = dt.datetime(1900, 1, 1)
  timesince = time - zero_time
  return timesince.seconds//60
    
def distance_matrix(df2):
  """returns the time/distance between any two nodes
  - creates a time/distance matrix between x/y coordinates
  - if both nodes are "real" pu/do nodes, read from the distance matrix
  - deals with special cases:
    - "infinite" time (very big time) to travel to source
    - zero time to travel from a source to any other node
    - zero time to travel from any node to a sink
    - "infinite" time big_num (very big time) to travel from a sink to any other node """

  # pixels per minute
  scaling_factor = 0.2

  dist = squareform(pdist(df2[['x','y']],metric = 'euclidean'))*scaling_factor

  # We return round(dist) due to problem instructions. Else, just return dist
  dist = np.round(dist)
  # We need to make sink/source (first and last indices) distance zero from every
  # other node
  # dist[i][j] is the distance from i to j
  
  # "infinitely big distance"
  big_dist = 2000
  for i in range(len(dist[0])):
    dist[i][0] = big_dist
    dist[0][i] = 0
    dist[-1][i] = big_dist
    dist[i][-1] = 0
  return np.round(dist)

def mins_to_time(mins):
  """turns mins back into HH:MM formated stringed"""
  h = int(mins//60)
  m = int(mins - h*60)
  if m > 10:
    return str(h)+ ":" + str(m)
  else:
    return str(h)+ ":0" + str(m)

"""# 3. Optimization problem setup and solving"""
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Overview of model variables:
# ----------------------------------
# The model will fit variables:
# x_ij = 1 iff a car uses edge (i,j) in a tour (zero otherwise)
# w_i = the time at when a car LEAVES node v_i (zero otherwise)
# y_ik = 1 iff car k visits node i (zero otherwise)
# It will have the following dummy variable:
# c_i = the number of people that are in the car that leaves node i

# and the problem is described using variables:
# d_i = 1 iff i is a pickup node / = -1 if i is a dropoff node / = 0 otherwise
# [E_i, L_i] - each node has an associated time window [E,L] 
# where E_i/L_i is the earliest/latest a car can leave node i
# t_ij is the time it takes to travel on edge (i,j)


def solve_ip(df2, K, silence_optimizer=True):
  """Defines and solves VRTW for K cars on input data df2"""
  
  m = Model()

  # Build t, distance matrix for df2 nodes
  t = distance_matrix(df2)

  # Helper function for indices
  # ----------------------------------

  def p_d(i):
    """returns the index for the dropoff of pickup node i"""
    # REQUIRES df2 - is called inside solver where df2 is defined
    return i + (len(df2)-2)//2

  # Iterators for loops
  # ----------------------------------
  
  # range for all nodes (df2 nodes + K artificial sinks)
  # (Note: we add K-1 because df2 already contains one copy of a sink node)
  N = range(len(df2)+K-1)
  # range of pickups
  N_p = range(1,len(df2)//2)
  # range of pickups
  N_d = range(len(df2)//2,len(df2)-1)
  # range of sinks
  N_sinks = range(len(df2)-1,len(df2)+K-1)
  # K becomes range of cars
  # We store num K as k_num
  k_num = K
  K = range(K)

  
  # VARIABLES to fit
  # ----------------------------------

  # binary variable indicating if edge (i,j) is used in car k's tour
  x = [[m.addVar(name = f"x{i},{j}", vtype="B") for j in N] for i in N]
  # binary variable indicating if car k leaves node i
  y = [[m.addVar(name = f"y{i},{k}", vtype="B") for k in K] for i in N]
  # variable indicating the time at which the vehicle leaves node i
  w = [m.addVar(name = f"w{i}", vtype="C") for i in N]
  # variable indicating the number of people when the vehicle leaves node i
  c = [m.addVar(name = f"c{i}", vtype="C") for i in N]


  # NO OBJECTIVE - (since we just want existance of a feasible solution)
  # ----------------------------------

  # CONSTRAINTS
  # ----------------------------------

  # Tour constraints
  # ----------------------------------

  # This ensures that each node is visited (in particular, a car leaves each node i) at least once:
  # For each node i, we need x[i][j] to be 1 for at least one j
  for i in N:
    m.addCons(quicksum(x[i][j] for j in N if j != i) >= 1)

  # This ensures that a subtour that enters i also leaves i
  for i in N:
    m.addCons(quicksum(x[a][i] for a in N if a !=i) - quicksum(x[i][b] for b in N if b !=i) == 0)

  # This ensures that, for a given trip request, the same car vists pickup and dropoff:
  # the dropoff of node i is j, where j = p_d(i)
  # and y_ik/y_jk = 1 iff car k visits node i/j respectively
  for i,k in product(N_p,K):
    m.addCons(y[i][k] - y[p_d(i)][k] == 0)

  big_num = 5000

  # This ensures that the car that enters i also leaves i
  # This constraint is a linearized version of the constraint:
  # (y_ik - y_jk)*x_ij = 0 (if edge (ij) is in a tour, then y_ik = y_jk)
  for i,j,k in product(N,N,K):
    # not for source nodes because they are special (all cars leave the source, node 0):
    if i != 0 and j != 0:
      m.addCons((y[i][k] - y[j][k]) - (1 - x[i][j])*big_num <= 0)
      m.addCons((y[j][k] - y[i][k]) - (1 - x[i][j])*big_num <= 0)
  
  # This ensures that each node (except for the sink, node 0) is served by one car:
  # i.e. that there is one k for which y_ik = 1
  for i in N:
    if i != 0:
      m.addCons(quicksum(y[i][k] for k in K) == 1)
    
  # This specifies that each car k leaves the source, node 0:
  for k in K:
    m.addCons(y[0][k] == 1)

  # This ensures that "sink node k" gets served by car k (one car exits "problem" at each sink)
  # (Note that N_sinks iterates over sink nodes)
  for k in K:
    for i in N_sinks:
      if i != len(df2)-1+k:
        m.addCons(y[i][k] == 0)
      else:
        m.addCons(y[i][k] == 1)


  # Time feasibility constraints
  # ----------------------------------

  # This ensures that a car never leaves a dropoff before its designated pickup
  # w_i is the time that car leaves node i, and node i's dropoff is p_d(i)
  for i in N_p:
    m.addCons(w[p_d(i)] >= w[i])

  # This ensures that all pickup and dropoff times must be within the appropirate time windows
  for i in N:
    # pickup/dropoff times for "real" (x,y) locations:
    if i < len(df2)-1:
      m.addCons(w[i] >= df2['E'][i])
      m.addCons(w[i] <= df2['L'][i])
    # pickup/dropoffs for sink nodes:
    else:
      m.addCons(w[i] >= df2['E'][len(df2)-1])
      m.addCons(w[i] <= df2['L'][len(df2)-1])

  # This is the "time limit" constraint:
  # This ensures that a car can't leave node j before getting to note j (when (i,j) is in a tour)
  # This constraint is a linearized version of the constraint:
  # x_ij*(w_i + t_ij - w_j) <= 0
  for i,j in product(N,N):
    # if i is a sink (i.e. i >= len(df2)-1):
    if i >= len(df2)-1:
      # sinks can only go back to source node 0
      if j != 0:
        m.addCons(x[i][j] == 0)
      # sinks can only go back to source node 0
      else:
        m.addCons(x[i][j] == 1)
    # if j is a sink (i.e. j >= len(df2)-1): t_ij = 0 (going to a sink is "free")
    elif j >= len(df2)-1:
      t_temp = 0 # can't travel back in time (t_temp here for human comprehension)
      m.addCons((w[i] + t_temp -w[j]) - (1-x[i][j])*big_num <= 0)
    else:
      m.addCons((w[i] + t[i][j] -w[j]) - (1-x[i][j])*big_num <= 0)

  
  # Car capacity constraints
  # ----------------------------------
  
  # c_i, the number of people in a car when it leaves node i, must be between 0 and 3:
  for i in N:
    m.addCons(c[i] <= 3)
    # no "negative" passengers, haha - no cheating IPs here!
    m.addCons(c[i] >= 0)

  # We know that there are 0 passengers in the car when it leaves the source
  m.addCons(c[0] == 0)

  # No car should arrive at a sink (i >= len(df2)-1) with people in it
  for i in N_sinks:
    m.addCons(c[i] == 0)

  # This ensures that c updates correctly when we pickup/dropoff people.
  # This constraint is a linearized version of the constraint:
  # x_ij*(c_i-c_j+d_j) <= 0
  # d_i = 1 iff i is a pickup node / = -1 if i is a dropoff node / = 0 otherwise
  # It ensures that when edge (i,j) is part of a tour, the difference
  # between c_i and c_j is +/- 1 (if i is a pickup/droppoff respectively)
  for i,j in product(N,N):
    d = 0
    # if j is a real node:
    if j > 0 and j < len(df2)-1:
      d = df2['d'][j]
    m.addCons((c[i] + d -c[j]) - (1-x[i][j])*4 <= 0)

    
  # EXTRA: We help the solver go faster by adding some values we know
  # ----------------------------------

  # All "back-edges" are zero.
  # i.e. no edges can go from a dropoff to its pickup
  for i in N_p:
    m.addCons(x[p_d(i)][i] == 0)

    # No edges can go between nodes who's time window is AHEAD and completely disjoint from anothers'
    # (These nodes might be in the tour of the same car, but the edges won't)
    for j in N_p: 
      if df2['L'][j] < df2['E'][i]:
        m.addCons(x[i][j] == 0)
        m.addCons(x[i][p_d(j)] == 0)
        m.addCons(x[p_d(i)][j] == 0)
        m.addCons(x[p_d(i)][p_d(j)] == 0)

  # OPTIMIZE!!!
  # ----------------------------------
  # if silence_optimizer = true (in main) we suppress all output from solver
  # (There's a lot of it, so I suppress it here by default.)
  m.hideOutput(silence_optimizer)
  m.optimize()
  return m


"""# 4. Make the deliverable (itinerary for humans to read) """
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------




# Make itinerary from solution:
# --------------------------------------------------------------------

def make_itinerary(m, df2, cars, path):
  """Function that produces an itinerary from a solved model object m
  ASSUMES that m.getStatus() == "optimal" 
  
  Takes in input SOLVED model m, df2, num of cars needed (cars),
  and path, the path to output file. Returns itinerary (as dataframe),
  prints itinerary, AND saves itinerary to path."""


  # 0. Extract values from m:

  d_sol = {}
  sol =  m.getBestSol()
  status = m.getStatus()
  print("status:", status)

  # Build d_sol:
  for v in m.getVars():
    if abs(sol[v]) > 0.5 :
      d_sol[v.name] = sol[v]
  
  # 1. Make a list of vertices visited per car:

  # vertices is a list of trips (one per car), where
  # each list in vertices is a list of stops (given as vertex ids), in the order they're visited
  vertices = []
  N = range(len(df2)+cars-1)
  N_p = range(1,len(df2)//2)

  for i in N_p:
    destinations = []
    # find locations j (nodes) visited by car i
    if f"x0,{i}" in d_sol:

      while i < len(df2)-1:
        for j in N:
          if f"x{i},{j}" in d_sol:
            destinations.append(i)
            i = j
      vertices.append(destinations)

  # 3. Make a nice output 

  # for each list in our list of vertices per car, make a nice output that includes
  # Dep time : w_i (the time at which car leaves this location to get to the next one)
  # X : x coordinate
  # Y : y coordinate
  # Description : pickup or dropoff
  # Trip: the trip number
  # Passenger: name of passenger

  # Helper structure to get the right text:
  type = ["!","Pickup", "Dropoff"]

  # itinerary is a list of dataframes - each dataframe is the schedule for a car
  car_stops = []

  for car_id, car in enumerate(vertices):
    for stop in car:
      # Build stop information:
      stop_details = []
      # add car number
      stop_details.append(car_id)
      # add dep time
      stop_details.append(mins_to_time(d_sol[f'w{stop}']))
      # add coordinates
      stop_details.append(df2.iloc[stop]['x'])
      stop_details.append(df2.iloc[stop]['y'])
      # add description
      stop_details.append(type[df2.iloc[stop]['d']])
      stop_details.append(df2.iloc[stop]['trip_n'])
      # add passenger name
      stop_details.append(df2.iloc[stop]['Requester'])      
      # Add stop information to car's stop list
      car_stops.append(stop_details)

  itinerary = pd.DataFrame(car_stops, columns = ['Vehicle','Dep time', "X", "Y", "Description", "Trip", "Passenger"])

  # 4. Produce output

  print(itinerary)
  itinerary.to_csv(path, index = False)

  return itinerary



"""# 5. MAIN """
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------

def main():

  # read in filename
  if len(sys.argv) < 2:
    # remind human to include file
    print(f"USAGE: python {sys.argv[0]} <input file>")
    sys.exit(-1)
  filename = sys.argv[1]

  # process data
  df = wrangle(filename)
  # Upper bound (assuming requester's rides are distinct) is the num of people
  k_upper = len(df['Requester'].unique())-2
  # Lower bound (before we begin searching)
  k_lower = 0
  m_best = None

  # Suppresses solver output if True
  silence_optimizer=True

  # binary search over possible values for solution (k = number of cars needed)
  while k_lower < k_upper-1:
    mid = (k_lower + k_upper)//2
    print(f"Initialized solving for {mid} cars...")
    m = solve_ip(df, mid, silence_optimizer)

    print(f"Finalized solving for {mid}. Solver status: {m.getStatus()}")

    # IP is feasible for k_upper:
    # IP not feasible for k_lower
    if m.getStatus() != "optimal":
      k_lower = mid
    else:
      k_upper = mid
      # we found a feasible sol:
      m_best = m
      schedule_path = f"schedule_for_{k_upper}_cars.csv"
      make_itinerary(m_best, df, k_upper, schedule_path)

  print(f"Feasible itinerary achieved with {k_upper} vehicles. No feasible solutions with fewer vehicles. Saving solution into {schedule_path}.")


if __name__ == "__main__":
    main()
