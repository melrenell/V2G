from __future__ import division
import datetime
import matplotlib.pyplot as plt
import v2gsim.model as model
import v2gsim.itinerary as itinerary
import v2gsim.post_simulation as post_simulation
import v2gsim.core as core
import pandas
import numpy as np
plt.rcParams.update({'font.size': 12})
# ### Require gurobi or CPLEX #####
opttype0="peak_shaving"
opttype1="penalized_peak_shaving"
opttype2="ramp_mitigation"

#inputs to change
opttype=opttype0
nb_vehicles=14000
nb_days=1 #be sure to also adjust the input load data
timeinterval=10 #10 minutes is default
charger_power=60000 #charger power in Watts
#itin='Schoolbus_25_summer_DCFC.xlsx'
#itinname='Schoolbus_25_summer_DCFC'
#loadname='Summer_avg' #for saved file
#loaddata='FERC_2018_Summer_Interpolate2.xlsx'
itin='Schoolbus_25_winter_DCFC.xlsx'
itinname='Schoolbus_25_winter_DCFC'
loadname='Winter_avg' #for saved file
loaddata='FERC_2018_Winter_Interpolate2.xlsx'
itinnumber=25
min_SOC=0.1
max_SOC=0.95

#V2G-Sim
project = model.Project()
project = itinerary.from_excel(project, itin)
project = itinerary.copy_append(project, nb_of_days_to_add=nb_days+1)

# This function from the itinerary module return all the vehicles that
# start and end their day at the same location (e.g. home)
project.vehicles = itinerary.get_cycling_itineraries(project)

# Reduce the number of vehicles
project.vehicles = project.vehicles[0:itinnumber]

# Create some new charging infrastructures, append those new
# infrastructures to the project list of infrastructures
charging_stations = []
charging_stations.append(
    model.ChargingStation(name='L2', maximum_power=7200, minimum_power=0))
charging_stations.append(
    model.ChargingStation(name='L1_V1G', maximum_power=1400, minimum_power=0, post_simulation=True))
charging_stations.append(
    model.ChargingStation(name='L2_V2G', maximum_power=charger_power, minimum_power=-charger_power, post_simulation=True))
project.charging_stations.extend(charging_stations)

# Create a data frame with the new infrastructures mix and
# apply this mix at all the locations
df = pandas.DataFrame(index=['L2', 'L1_V1G', 'L2_V2G'],
                      data={'charging_station': charging_stations,
                            'probability': [0.0, 0.0, 1.0]})
for location in project.locations:
    if location.category in ['Work', 'Home']:
        location.available_charging_station = df.copy()

# Initiate SOC and charging infrastructures
core.initialize_SOC(project, nb_iteration=2)

# Assign a basic result function to save power demand
for vehicle in project.vehicles:
    vehicle.result_function = post_simulation.netload_optimization.save_vehicle_state_for_optimization
    
# Launch the simulation
core.run(project, date_from=project.date + datetime.timedelta(days=1),
                date_to=project.date + datetime.timedelta(days=nb_days+1),
                reset_charging_station=False)  
    
# Look at the results
total_power_demand = post_simulation.result.total_power_demand(project)

# Optimization
myopti = post_simulation.netload_optimization.CentralOptimization(project, timeinterval,
                                                                         project.date + datetime.timedelta(days=1),
                                                                         project.date + datetime.timedelta(days=nb_days+1),
                                                                         minimum_SOC=min_SOC, maximum_SOC=max_SOC)
# Load the net load data
finalResult = pandas.DataFrame()
net_load=pandas.read_excel(loaddata)
net_load= pandas.DataFrame(net_load['netload'])
i = pandas.date_range(start=project.date + datetime.timedelta(days=1), 
                      end=project.date + datetime.timedelta(days=nb_days+1),
                      freq='T', closed='left')
net_load = net_load.set_index(i)
net_load = net_load.resample(str(project.timestep) + 'S')
net_load = net_load.fillna(method='ffill').fillna(method='bfill')
myresult = myopti.solve(project, net_load* 1000000,
                        nb_vehicles, peak_shaving=opttype, SOC_margin=0.05) 

unoptimized_demand = pandas.DataFrame()
unoptimized_charging = pandas.DataFrame()
optimized_charging = pandas.DataFrame()
net_loads = net_load.copy()
total_power_demand_without_opti = []
net_load_updated = net_load.copy()
total_power_demand_without_opti.append(post_simulation.result.total_power_demand(project)['total'])
index=0
nb_project=1 

#Store results
unoptimized_demand["Proj_"+str(index)] = total_power_demand_without_opti[index]
unoptimized_charging["Proj_"+str(index)] = myresult['vehicle_before']
optimized_charging["Proj_"+str(index)] = myresult['vehicle_after']
net_loads["Proj_"+str(index)] = net_load_updated['netload']

scale_factor = nb_vehicles / (nb_project * len(project.vehicles))/1000000
temp_load = np.array(net_load_updated['netload'])
temp_opt = np.array(myresult['vehicle_after']) * scale_factor
temp_unopt_dem = np.array(total_power_demand_without_opti[index]) * scale_factor
temp_unopt_chg = np.array(myresult['vehicle_before']) * scale_factor

net_load_updated['netload'] =  temp_opt + temp_load #+ temp_unopt_dem - temp_unopt_chg

#Saving unoptimized demand
unopt_dem_tosave = unoptimized_demand.copy()
unopt_dem_tosave = unopt_dem_tosave * scale_factor
unopt_dem_tosave.index = net_load.index

netload = np.array(net_load['netload'])
unopt_dem = np.array(unoptimized_charging.sum(axis=1)) * scale_factor
unopt = np.array(unoptimized_demand.sum(axis=1)) * scale_factor
opt = np.array(optimized_charging.sum(axis=1)) * scale_factor
results = pandas.DataFrame.from_dict({"net_load": netload,
              'Unoptimized_charging': unopt,
              'Optimized_charging': opt,
              'Net_load_unoptimized': netload + unopt,
              'Net_load_optimized': netload + opt})
results.index = net_load.index

results.to_csv(str(loadname)+str(opttype)+str(itinname)+str(nb_vehicles)+".csv")

fig = plt.figure(figsize=(15,5))
plt.plot(results['net_load'], label='net_load')
plt.plot(results['Net_load_optimized'], label='net_load_optimized')
plt.plot(results['Net_load_unoptimized'], label='net_load_unoptimized',color='m')
plt.legend()
plt.grid(b=None)
plt.ylabel('Power [MW]')
plt.xlabel('Time of Day')
plt.title('Daily Load Profile')
