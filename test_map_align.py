import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import math
import matplotlib.transforms as mtransforms

def add_rotated_basemap(ax, origin_lat, origin_lon, heading_deg, zoom=19):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # 2. Get GPS bounding box to fetch image
    corners_local = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    
    METERS_PER_DEG_LAT = 111320
    METERS_PER_DEG_LON = 111320 * math.cos(math.radians(origin_lat))
    
    lats, lons = [], []
    for x, y in corners_local:
        heading_rad = math.radians(heading_deg)
        north_offset = x * math.cos(heading_rad) + y * math.sin(heading_rad)
        east_offset = x * math.sin(heading_rad) - y * math.cos(heading_rad)
        
        lats.append(origin_lat + (north_offset / METERS_PER_DEG_LAT))
        lons.append(origin_lon + (east_offset / METERS_PER_DEG_LON))
        
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Fetch map
    img, ext = ctx.bounds2img(min_lon, min_lat, max_lon, max_lat, zoom=zoom, source=ctx.providers.OpenStreetMap.Mapnik, ll=True)
    
    # Calculate Web Mercator for origin
    r = 6378137.0
    mx0 = r * math.radians(origin_lon)
    my0 = r * math.log(math.tan(math.pi/4 + math.radians(origin_lat)/2))
    
    scale = math.cos(math.radians(origin_lat))
    
    # Build transformation: Image (Web Mercator) -> Local (LiDAR coords)
    t = mtransforms.Affine2D()
    t.translate(-mx0, -my0)
    t.scale(scale, scale)
    
    h_rad = math.radians(heading_deg)
    matrix = np.array([
        [math.sin(h_rad), -math.cos(h_rad), 0],
        [math.cos(h_rad), math.sin(h_rad), 0],
        [0, 0, 1]
    ])
    # WAIT! The inverse of:
    # North = X * cos + Y * sin
    # East = X * sin - Y * cos
    # [North] = [cos sin ] [X]
    # [East ]   [sin -cos] [Y]
    # Determ = -cos^2 - sin^2 = -1
    # Inverse:
    # 1/(-1) * [-cos -sin]
    #          [-sin  cos]
    # = [cos  sin]
    #   [sin -cos]
    # So X = North * cos + East * sin
    # Y = North * sin - East * cos
    # Let us build the correct matrix mapping (East, North) -> (X, Y)
    # [X] = [sin cos] [East ]
    # [Y] = [-cos sin] [North]
    matrix = np.array([
        [math.sin(h_rad), math.cos(h_rad), 0],
        [-math.cos(h_rad), math.sin(h_rad), 0],
        [0, 0, 1]
    ])
    t += mtransforms.Affine2D(matrix)
    
    # Add data transform
    ax.imshow(img, extent=ext, transform=t + ax.transData, zorder=0)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
add_rotated_basemap(ax, 51.045101, 3.713908, 19.4, zoom=18)
ax.scatter(0, 0, color="red", marker="^", s=200, zorder=10)
plt.savefig("test_map_transform.png")

