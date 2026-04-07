#########################################################################################################
# IMPORTS
#########################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#########################################################################################################
# LOAD DATA
# Data converted from json to csv using another python code.
# 
#########################################################################################################
print("Started...")

#Load data from csv file.. which was converted from json file(json to csv conversion was in a seperate python code)
df = pd.read_csv('C:/Dir/IITM-M.Tech/Wireless Communication/MiniProject/zone_cell_data/zone2.csv')

#Dropping neighbour data as cell phone is not connected to it. 
#It can create noise due to reflections
#Cell Phone only connected to the registered tower
#So to calculate path loss model we can are using registered type in our data
reg = df[df['Type'] == 'registered'].copy()
#########################################################################################################
# ANALYZE DATA
#########################################################################################################

#Checking spread of data using histogram
plt.figure()
plt.hist(reg['rsrp'], bins=30)
plt.xlabel("SNR (rsrp)")
plt.ylabel("Frequency")
plt.title("Histogram of SNR")
plt.savefig("C:/Dir/IITM-M.Tech/Wireless Communication/MiniProject/zone_cell_data/Histogram of SNR.png")
plt.show()

#Removing rssnr as maximum values are 2147483647 which is int32 max value
#Removing rsrp also has some maximum values are 2147483647 which is int32 max value so dropping it before building models
# reg = reg[reg['rsrp']  != 2147483647]
# reg = reg[reg['rssnr'] != 2147483647]

reg = reg[(reg['rsrp'] >= -140) & (reg['rsrp'] <= -44)].copy()

#Checking again spread of data using histogram
#Checking spread of data using histogram
plt.figure()
plt.hist(reg['rsrp'], bins=30)
plt.xlabel("SNR (rsrp)")
plt.ylabel("Frequency")
plt.title("Filtered SNR Histogram")
plt.savefig("C:/Dir/IITM-M.Tech/Wireless Communication/MiniProject/zone_cell_data/Filtered SNR Histogram.png")
plt.show()

#Analyzing and Plotting lat Long data to check for any outliers 
# print(reg['Lat'].describe())
# print(reg['Long'].describe())
plt.figure()
plt.scatter(reg['Long'], reg['Lat'])#, color='blue', marker='o')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(80.237700, 80.241800)
plt.ylim(12.992180, 13.004300)
plt.title("Location")
plt.grid(True)
plt.savefig("C:/Dir/IITM-M.Tech/Wireless Communication/MiniProject/zone_cell_data/Location.png")
plt.show()


#########################################################################################################
# DISTANCE
#########################################################################################################
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1)/2)**2 +
         np.cos(phi1)*np.cos(phi2)*np.sin(np.radians(lon2 - lon1)/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))


# Assuming Base location from a reading which is near to the tower
bs_locs = reg.groupby('pci')[['Lat','Long']].mean().rename(columns={'Lat':'bs_lat','Long':'bs_lon'}).reset_index()
reg = reg.merge(bs_locs, on='pci')

#Calulating distance of each row from the base location(lat, long)
reg['distance_m'] = haversine(reg['Lat'], reg['Long'], reg['bs_lat'], reg['bs_lon'])

#Avoided nearby signals 
reg = reg[(reg['distance_m'] > 20) & (reg['distance_m'] < 3000)]

#########################################################################################################
# PATH LOSS
#########################################################################################################
#Assuming Power sent by Transmitter
P_TX = 58.0

#Calculating mathematical path loss
reg['path_loss_db'] = P_TX - reg['rsrp']

#########################################################################################################
# FEATURES
#########################################################################################################
# LOG-DISTANCE MODEL (mean path loss, no shadowing) 
#   Mathematical Formulae PL(d) = PL0 + 10*n*log10(d)
#   Calculate change in lat and lon from base tower lat long
#   Calculate Angle to the tower using delta lat long
reg['log_distance'] = np.log10(reg['distance_m'])
reg['delta_lat'] = reg['Lat'] - reg['bs_lat']
reg['delta_lon'] = reg['Long'] - reg['bs_lon']
reg['angle'] = np.arctan2(reg['delta_lat'], reg['delta_lon'])

#########################################################################################################
# CLUSTERING
#########################################################################################################
#Applying k-means cluster algo
scaler = StandardScaler()
scaled = scaler.fit_transform(reg[['distance_m','path_loss_db']])
reg['cluster'] = KMeans(n_clusters=2, random_state=42).fit_predict(scaled)

#########################################################################################################
# LOG DISTANCE
#########################################################################################################
log_d = reg['log_distance'].values
pl = reg['path_loss_db'].values

X_fit = np.column_stack([np.ones_like(log_d), log_d])
pl0_fit, n10 = np.linalg.lstsq(X_fit, pl, rcond=None)[0]
n_fit = n10 / 10

#########################################################################################################
# DATA SPLIT
#########################################################################################################
# Features list, X,Y and Splitting dataset
features = ['log_distance','delta_lat','delta_lon','angle','cluster']
X = reg[features].values
y = reg['path_loss_db'].values

y_ld = pl0_fit + 10*n_fit*reg['log_distance']
residual = y - y_ld

X_train, X_test, y_train, y_test, res_train, res_test = train_test_split(
    X, y, residual, test_size=0.2, random_state=42
)

#########################################################################################################
# MODELS
#########################################################################################################

# Random Forest Algo.
rf = RandomForestRegressor(n_estimators=200, max_depth=12)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# XGBoost Algo.
xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Hybrid RF 
rf_res = RandomForestRegressor(n_estimators=200, max_depth=12)
rf_res.fit(X_train, res_train)
y_pred_hybrid_rf = (pl0_fit + 10*n_fit*X_test[:,0]) + rf_res.predict(X_test)

# Hybrid XGB
xgb_res = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
xgb_res.fit(X_train, res_train)
y_pred_hybrid_xgb = (pl0_fit + 10*n_fit*X_test[:,0]) + xgb_res.predict(X_test)

# Log-distance
y_pred_ld = pl0_fit + 10*n_fit*X_test[:,0]

#########################################################################################################
# COMPARISON METRICS
#########################################################################################################
def evaluate(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred)), r2_score(y_true,y_pred)

models = {
    "Log-Distance": y_pred_ld,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb,
    "Hybrid RF": y_pred_hybrid_rf,
    "Hybrid XGB": y_pred_hybrid_xgb
}

print("\nMODEL COMPARISON\n")
for name, pred in models.items():
    rmse, r2 = evaluate(y_test, pred)
    print(f"{name:<15} RMSE: {rmse:.2f}   R²: {r2:.4f}")

#########################################################################################################
# GENERATE PLOTS
#########################################################################################################
plt.figure(figsize=(10,6))

samp = reg.sample(min(5000,len(reg)))
plt.scatter(samp['distance_m'], samp['path_loss_db'], s=5, alpha=0.2)

d_curve = np.linspace(reg['distance_m'].min(), reg['distance_m'].max(), 500)
ld_curve = np.log10(d_curve)

plt.plot(d_curve, pl0_fit + 10*n_fit*ld_curve, label='Log-Distance')
plt.plot(d_curve, xgb.predict(np.column_stack([ld_curve, np.zeros((500,4))])), label='XGBoost')

plt.legend()
plt.xlabel("Distance")
plt.ylabel("Path Loss")
plt.grid()

plt.savefig("C:/Dir/IITM-M.Tech/Wireless Communication/MiniProject/zone_cell_data/final_comparison_plot.png")
print("Plot saved.")
#########################################################################################################
