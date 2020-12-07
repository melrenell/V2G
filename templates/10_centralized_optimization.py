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
# Create a project and initialize it with someitineraries

#inputs to change
opttype0="peak_shaving"
opttype1="penalized_peak_shaving"
opttype2="ramp_mitigation"
opttype=opttype0
nb_vehicles=14000
nb_days=1 #be sure to also adjust the input load data
timeinterval=10 #10 is default
itin='Schoolbus_25_summer_DCFC.xlsx'
itinname='Schoolbus_25_summer_DCFC'
loadname='Summer_avg' #for saved file
charger_power=60000 #charger power in Watts
#loaddata='FERC_2018_Summer_Interpolate2.xlsx'
loaddata='FERC_2018_Summer_Interpolate2.xlsx'



#V2G-Sim
project = model.Project()
project = itinerary.from_excel(project, itin)
project = itinerary.copy_append(project, nb_of_days_to_add=nb_days+1)

# This function from the itinerary module return all the vehicles that
# start and end their day at the same location (e.g. home)
project.vehicles = itinerary.get_cycling_itineraries(project)

# Reduce the number of vehicles
project.vehicles = project.vehicles[0:25]

# Create some new charging infrastructures, append those new
# infrastructures to the project list of infrastructures
charging_stations = []
charging_stations.append(
    model.ChargingStation(name='L2_V2G', maximum_power=charger_power, minimum_power=-charger_power, post_simulation=True))
project.charging_stations.extend(charging_stations) 

# Create a data frame with the new infrastructures mix and
# apply this mix at all the locations
df = pandas.DataFrame(index=['L2_V2G'],
                      data={'charging_station': charging_stations,
                            'probability': [1.0]}) #Hard-coded to assume only V2G
for location in project.locations:
    if location.category in ['Work']:
        location.available_charging_station = df.copy() #Hard-coded to assume only V2G

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
#v2gsim.post_simulation.netload_optimization.CentralOptimization(project, optimization_timestep, 
                                                                #date_from, date_to, 
                                                                #minimum_SOC=0.1, maximum_SOC=0.95)
myopti = post_simulation.netload_optimization.CentralOptimization(project, timeinterval,
                                                                         project.date + datetime.timedelta(days=1),
                                                                         project.date + datetime.timedelta(days=nb_days+1),
                                                                         minimum_SOC=0.1, maximum_SOC=0.95)
# Load the net load data
finalResult = pandas.DataFrame()









net_load=pandas.read_excel(loaddata)
net_load= pandas.DataFrame(net_load['netload'])
print('Original Net Load')
print(net_load)
i = pandas.date_range(start=project.date + datetime.timedelta(days=1), 
                      end=project.date + datetime.timedelta(days=nb_days+1),
                      freq='T', closed='left')
net_load = net_load.set_index(i)
print(net_load)
net_load = net_load.resample(str(project.timestep) + 'S')
net_load = net_load.fillna(method='ffill').fillna(method='bfill')
myresult = myopti.solve(project, net_load* 1000000,#Netload in Watts
                        nb_vehicles, peak_shaving=opttype, SOC_margin=0.05) 









##############################################
#Margaret's Code
##############################################
import pandas as pd
unoptimized_demand = pd.DataFrame()
unoptimized_charging = pd.DataFrame()
optimized_charging = pd.DataFrame()
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
#unopt_dem_tosave.to_csv(results_path+"/unoptimized_demand.csv")

#Combining results
#opti_results_path = current_path + '/Opti_Results/'+ foldername
#if not os.path.exists(opti_results_path):
#    os.makedirs(opti_results_path)

netload = np.array(net_load['netload'])
unopt_dem = np.array(unoptimized_charging.sum(axis=1)) * scale_factor
unopt = np.array(unoptimized_demand.sum(axis=1)) * scale_factor
opt = np.array(optimized_charging.sum(axis=1)) * scale_factor
results = pd.DataFrame.from_dict({"net_load": netload,
              'Unoptimized_charging': unopt,
              'Optimized_charging': opt,
              'Net_load_unoptimized': netload + unopt,
              'Net_load_optimized': netload + opt})
results.index = net_load.index

#dd = datetime.datetime.now().day
#mm = datetime.datetime.now().month

###########################results.to_csv(str(loadname)+str(opttype)+str(itinname)+str(nb_vehicles)+".csv")

#Plotting results
#fig = plt.figure(figsize=(15,5))
#plt.plot(results['net_load'], label='net_load')
#plt.plot(results['Optimized_charging'], label='optimized charging demand')
#plt.plot(results['Unoptimized_charging'], label='unoptimized charging demand')
#plt.legend()
#ticks0=np.arange(0,1440,60)
#ticks1=np.arange(0,24,1)
#ticks0=np.arange(0,2880,60)
#ticks1=np.arange(0,48,1)
#ticks0=np.arange(0,5760,60)
#ticks1=np.arange(0,96,1)
#plt.ylabel('Power (MW)')
#plt.xlabel('Time of day')
#plt.xticks(ticks0,ticks1)
#plt.title('Bus itinerary, NC Summer net load, 1.5 mil vehicles')
#nb_vehicles=np.int(nb_vehicles)
fig = plt.figure(figsize=(15,5))
plt.plot(results['net_load'], label='net_load')
plt.plot(results['Net_load_optimized'], label='net_load_optimized')
plt.plot(results['Net_load_unoptimized'], label='net_load_unoptimized',color='m')
plt.legend()
ticks0=np.arange(0,1440,60)
ticks1=np.arange(0,24,1)
#ticks0=np.arange(0,2880,60)
#ticks1=np.arange(0,48,1)
#ticks0=np.arange(0,5760,60)
#ticks1=np.arange(0,96,1)
plt.text(10,15000,'Optimization Type = %s' %opttype)
plt.text(10,14800,'Number of vehicles = %s' %nb_vehicles)
plt.text(10,14600,'Time interval = %s minutes' %timeinterval)
plt.ylabel('Power (MW)')
plt.xlabel('Time of day')
#plt.xticks(ticks0,ticks1)
plt.title('Bus itinerary, NC Summer net load')
"""

###############################################
#Ends here
###############################################

# Get the result in the right format
temp_vehicle = pandas.DataFrame(
    (total_power_demand['total'] - myresult['vehicle_before'] + myresult['vehicle_after']) *
    (nb_vehicles / len(project.vehicles)) / (1000 * 1000)) #netload in MW  
temp_vehicle = temp_vehicle.rename(columns={0: 'vehicle'})
temp_vehicle['index'] = range(0, len(temp_vehicle))
temp_vehicle = temp_vehicle.set_index(['index'], drop=True)

temp_netload = net_load.copy()
#temp_netload = temp_netload.resample('60S')
temp_netload = temp_netload.fillna(method='ffill').fillna(method='bfill')
temp_netload = temp_netload.head(len(temp_vehicle))
tempIndex = temp_netload.index
temp_netload['index'] = range(0, len(temp_vehicle))
temp_netload = temp_netload.set_index(['index'], drop=True)

temp_result = pandas.DataFrame(temp_netload['netload'] + temp_vehicle['vehicle'])
temp_result = temp_result.rename(columns={0: 'netload'})
temp_result = temp_result.set_index(tempIndex)
temp_netload = temp_netload.set_index(tempIndex)
temp_vehicle = temp_vehicle.set_index(tempIndex)
#temp_result.to_csv("Results1014.csv")
#temp_netload.to_csv("Netload1014.csv")

#Adjust x-axis scale
ticks0=np.arange(0,1440,60)
ticks1=np.arange(0,24,1)
plt.figure()
plt.plot(temp_netload['netload'], label='netload')
plt.plot(temp_result['netload'], label='netload + vehicles')
#plt.plot(results['Net_load_unoptimized'], label='net_load_unoptimized',color='m')
plt.title('Summer 2018 NC netload w/ 14,000 V2G Leafs')
plt.ylabel('Power (MW)')
plt.xlabel('Time of day')
plt.xticks(ticks0,ticks1)
plt.legend()
plt.show()

#import pdb
#pdb.set_trace()
"""