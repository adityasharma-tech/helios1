import matplotlib.pyplot as plt
import numpy as np

# Coordinates (Upper Left, Upper Right, Lower Right, Lower Left)
lats = [-84.795295, -84.841821, -84.175074, -84.133675]
lons = [35.905091, 34.891890, 31.793314, 32.714057]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

# Convert lat/lon to polar coordinates for South Pole
# Theta = longitude in radians
# R = 90 - abs(latitude)
theta = np.deg2rad(lons)
r = 90 - np.abs(lats)

# Close the polygon
theta = np.append(theta, theta[0])
r = np.append(r, r[0])

ax.plot(theta, r, color='red', linewidth=2, marker='o', label='CH2 Image Footprint')
ax.fill(theta, r, color='red', alpha=0.3)

ax.set_ylim(0, 10) # 80S to 90S
ax.set_yticks(np.arange(0, 11, 2))
ax.set_yticklabels(['-90°', '-88°', '-86°', '-84°', '-82°', '-80°'])

plt.title("CH2 Image Footprint near the Lunar South Pole")
plt.legend()
plt.savefig("/home/friday/helios1/footprint_plot.png", dpi=300)
print("Saved to /home/friday/helios1/footprint_plot.png")
