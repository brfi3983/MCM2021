import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kde

# import geopandas as gpd
# from mpl_toolkits.basemap import Basemap
plt.style.use('ggplot')

# ========================================================
def main():
	# Import Main Data
	df = pd.read_csv('2021MCMProblemC_DataSet.csv')

	# Convert to workable dates
	df['Detection Date'] = pd.to_datetime(df['Detection Date'], errors = 'ignore', dayfirst=True)
	df['Submission Date'] = pd.to_datetime(df['Submission Date'], dayfirst=True)

	# Sort by dates and grab label, lat, and long
	evolution = df.sort_values(by='Detection Date')
	evolution = evolution[['Lab Status', 'Latitude', 'Longitude']]

	# Map label to color for graphing
	evolution['Lab Status'] = evolution['Lab Status'].replace({'Positive ID': 'red', 'Negative ID': 'green', 'Unverified': 'pink', 'Unprocessed': 'yellow'})

	# Plot each time step
	plt.title('Evolution of Sightings')
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')


	for n in range(100, evolution.shape[0]):

		if n % 100 == 0:
			t = evolution.head(n)
			t = np.array(t)[:,1:]
			x = t[:, 1]
			y = t[:,0]
			print(x,y)
			# plt.scatter(x=t['Longitude'], y=t['Latitude'], c=t['Lab Status'])
			# plt.hist2d(t[:,1], t[:,0], bins=100)
			# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
			nbins=300
			k = kde.gaussian_kde([x,y])
			xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
			zi = k(np.vstack([xi.flatten(), yi.flatten()]))

			# Make the plot
			plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
# plt.show()

			# sns.kdeplot(data=t, x=t[:, 1], y=t[:, 0], fill=True, thresh=0, levels=100, cmap="mako",)
			plt.show()
			exit()
			plt.savefig(f'gif/{n}.png')


	# fp = "C:/Users/user/Downloads/pacificnorthwestregionalwindhighresolution/pnw_50mwindnouma.shp"
	# data = gpd.read_file(fp)
	# plt.show()

# ========================================================
if __name__ == "__main__":
	main()