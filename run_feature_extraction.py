import os
import time
import json
import fiona
import torch
import urllib
import requests
import warnings
import webbrowser
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
from tqdm import tqdm
from itertools import cycle
from datetime import timedelta
from pykalman import KalmanFilter

import datetime
from pathlib import Path

import folium
from folium import Map
from folium import plugins

from tqdm.contrib import tzip

import geopandas
from geopandas import GeoSeries

from shapely import wkt
from shapely.wkt import loads
from shapely.ops import transform
from shapely.geometry import shape
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.geometry import mapping
from shapely.geometry import Polygon
from shapely.geometry import LineString

from Datasets.Geolife.read_geolife import read_all_users_geolife
from Datasets.Geolife.read_geolife import read_geolife_gps
from Datasets.iWet.read_iwet import read_all_iwet, read_iwet_gps

from matplotlib import cm
import matplotlib.pyplot as plt

from fmm import STMATCH, STMATCHConfig
from fmm import FastMapMatch, Network, NetworkGraph, UBODTGenAlgorithm, UBODT, FastMapMatchConfig

from scipy.spatial.distance import cdist

from geopy.distance import geodesic
from geopy.distance import distance as geodist


class WrongNumberOfLegs(Exception):
    pass


"""
References: 

[1] Cordonnier, J. B., & Loukas, A. (2019). Extrapolating paths with graph neural networks.
 arXiv preprint arXiv:1903.07518.
 
[2] Yang, C., & Gidofalvi, G. (2018). Fast map matching, an algorithm integrating hidden Markov model 
with precomputation. International Journal of Geographical Information Science, 32(3), 547-570. 
Weblink: https://fmm-wiki.github.io/

[3] Yu Zheng, Xing Xie, Wei-Ying Ma, GeoLife: A Collaborative Social Networking Service among User, 
location and trajectory. Invited paper, in IEEE Data Engineering Bulletin. 33, 2, 2010, pp. 32-40.
weblink: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

"""


def graph_shapefile_directional(G, dataset_name, save=False, encoding="utf-8"):
    if not save:
        return

    filepath = os.path.join('Datasets', dataset_name, 'meta', 'graphInfo')

    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    # We need a unique ID for each edge
    gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype='int')
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)

    check_graph = False
    if check_graph:
        gdf_nodes = geopandas.read_file(os.path.join("Datasets", dataset_name, "meta", 'graphInfo', "nodes.shp"),
                                        layer='nodes').set_index('osmid')
        gdf_edges = geopandas.read_file(os.path.join("Datasets", dataset_name, "meta", 'graphInfo', "edges.shp"),
                                        layer='edges').set_index(['u', 'v', 'key'])
        assert gdf_nodes.index.is_unique and gdf_edges.index.is_unique

        # graph_attrs = {'crs': 'epsg:4326', 'simplified': True}
        graph_attrs = {'simplified': False}
        _G = ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs)
        mapped_edges_id = dict(((fr, to), data) for i, (fr, to, data) in enumerate(G.edges(data=True)))
        _mapped_edges_id = dict(((fr, to), data) for i, (fr, to, data) in enumerate(_G.edges(data=True)))
        for key, value in mapped_edges_id.items():
            if mapped_edges_id[key]['osmid'] != mapped_edges_id[key]['osmid']:
                assert ValueError('Not matched edge')

        mapped_nodes_id = dict((i, data) for i, data in G.nodes(data=True))
        _mapped_nodes_id = dict((i, data) for i, data in _G.nodes(data=True))
        for key, value in mapped_nodes_id.items():
            if _mapped_nodes_id[key]['y'] != _mapped_nodes_id[key]['y'] \
                    or _mapped_nodes_id[key]['x'] != _mapped_nodes_id[key]['x']:
                assert ValueError('Not matched node')


def map_graph(latMin, latMax, longMin, longMax, dataset_name, download=False, plt_graph=False):
    filepath = 'Datasets/{}/meta/graphInfo'.format(dataset_name)
    ox.config(use_cache=True, log_console=True)
    if download:
        print('Download graph...')
        start_time = time.time()
        bounds = (longMin, longMax, latMin, latMax)
        x1, x2, y1, y2 = bounds
        boundary_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        # The street networks are directed and preserve one-way directionality.
        G = ox.graph_from_polygon(boundary_polygon, network_type='drive', simplify=False)

        # G, _ = ox.graph_from_address("Thessaloniki, Greece", dist=10000, return_coords=True,
        #                              network_type='drive', simplify=False)
        # # Add missing edge speeds
        G = ox.speed.add_edge_speeds(G)
        ox.save_graphml(G, filepath=os.path.join(filepath, '{}.graphml'.format(dataset_name)))

        print("--- %s seconds to download graph---" % (time.time() - start_time))
    else:
        print('load graph...')
        start_time = time.time()
        G = ox.load_graphml(os.path.join(filepath, '{}.graphml'.format(dataset_name)))
        print("--- %s seconds to load graph---" % (time.time() - start_time))

    if plt_graph:
        # ox.plot_graph(G)

        location = [latMin / 2 + latMax / 2, longMin / 2 + longMax / 2]
        map_osm = Map(location=location,
                      min_lat=latMin,
                      max_lat=latMax,
                      min_lon=longMin,
                      max_lon=longMax,
                      zoom_start=11,
                      use_local_extrema=False, tiles="Stamen Terrain")
        upper_left = (latMax, longMin)
        upper_right = (latMax, longMax)
        lower_right = [latMin, longMax]
        lower_left = (latMin, longMin)

        bounds = [upper_left, upper_right, lower_right, lower_left]

        folium.Rectangle(bounds=bounds, color='#ff7800',
                         fill=True, fill_color='#ffff00',
                         fill_opacity=0.2).add_to(map_osm)
        folium.Marker([latMin, longMin], color='blue').add_to(map_osm)
        folium.Marker([latMax, longMax], color='red').add_to(map_osm)

        Path(f"{timestamp}/viz_debug/{dataset_name}").mkdir(parents=True, exist_ok=True)

        map_osm.save(f"viz_debug/{dataset_name}/{timestamp}/{dataset_name}.html")
        webbrowser.open(f"viz_debug/{dataset_name}/{timestamp}/{dataset_name}.html")

    return G


def shortest_path_distance_matrix(G, dataset_name, save=False):
    if not save:
        return

    filepath = os.path.join('Datasets', dataset_name, 'meta', 'graphInfo')

    # G = nx.path_graph(50) # example graph

    # takes some time...
    print('Save shortest-path-distance-matrix.pt')
    # G = nx.convert_node_labels_to_integers(G)
    pairwise_distances = dict(((starter, target), length)
                              for starter, target_dict in tqdm(nx.all_pairs_shortest_path_length(G, cutoff=30))
                              for target, length in target_dict.items())

    sparse_storage = True
    if sparse_storage:
        # # Sparse storage
        row = []
        col = []
        data = []
        for (fr, to), dist in tqdm(pairwise_distances.items()):
            row.append(fr)
            col.append(to)
            data.append(dist)

        # from scipy import sparse
        # pairwise_dist = sparse.coo_matrix((data, (row, col)), shape=(len(G.nodes), len(G.nodes)))

        i = torch.tensor(np.vstack((row, col)))
        v = torch.tensor(data, dtype=torch.int8)
        pairwise_dist = torch.sparse_coo_tensor(i, v, [len(G.nodes), len(G.nodes)])
    else:
        # # # Dense storage
        # # save pairwise distances
        pairwise_dist = torch.zeros([len(G.nodes), len(G.nodes)], dtype=torch.int8) - 1
        for (fr, to), dist in tqdm(pairwise_distances.items()):
            pairwise_dist[fr, to] = dist

    # np.save(os.path.join(filepath, 'shortest-path-distance-matrix.npy'), pairwise_dist)
    torch.save(pairwise_dist, os.path.join(filepath, 'shortest-path-distance-matrix.pt'))


def graph_node_edges(G, dataset_name, save=False):
    if not save:
        return

    # impute missing edge speeds and calculate edge travel times with the speed module
    # # # G = ox.speed.add_edge_travel_times(G)
    filepath = os.path.join('Datasets', dataset_name, 'meta', 'graphInfo')
    # save nodes
    print('Save nodes.txt...')
    filename = os.path.join(filepath, 'nodes.txt')
    with open(filename, 'w') as f:
        f.write("{}\t{}\n".format(len(G.nodes), 2))  # num_nodes, num_node_features
        for i, node in G.nodes.data():
            line = "\t".join(map(str, [i, node['x'], node['y']])) + "\n"
            f.write(line)

    # save edges
    print('Save edges.txt...')
    filename = os.path.join(filepath, 'edges.txt')
    with open(filename, 'w') as f:
        f.write("{}\t{}\n".format(len(G.edges), 2))
        for i, (fr, to, edge) in enumerate(G.edges.data()):
            maxspeed = edge['speed_kph'] if 'speed_kph' in edge else -1
            length = edge['length'] if 'length' in edge else -1

            line = "\t".join(map(str, [i, fr, to, length, maxspeed])) + "\n"
            f.write(line)


def get_road_graph(latMin, latMax, longMin, longMax, dataset_name):
    filepath = 'Datasets/{}/meta'.format(dataset_name)
    # from box
    print('Download graph...')
    bounds = (longMin, longMax, latMin, latMax)
    x1, x2, y1, y2 = bounds
    boundary_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    G = ox.graph_from_polygon(boundary_polygon, network_type='drive', simplify=False)
    start_time = time.time()
    graph_shapefile_directional(G, dataset_name=dataset_name)
    print("--- %s seconds ---" % (time.time() - start_time))

    # From xml
    # graph = ox.load_graphml('Datasets/Beijing_xml/box_Beijing.graphml')
    # graph_shapefile_directional(graph, filepath='Datasets/Beijing_xml/box_Beijing', encoding="utf-8")

    # takes some time...
    print('Save shortest-path-distance-matrix.npy')
    G = nx.convert_node_labels_to_integers(G)
    pairwise_distances = dict(((starter, target), length)
                              for starter, target_dict in tqdm(nx.all_pairs_shortest_path_length(G, cutoff=30))
                              for target, length in target_dict.items())

    # save pairwise distances
    pairwise_dist = np.zeros([len(G.nodes), len(G.nodes)], dtype=np.dtype(np.uint8)) - 1
    for (fr, to), dist in tqdm(pairwise_distances.items()):
        pairwise_dist[fr, to] = dist

    np.save(os.path.join(filepath, 'shortest-path-distance-matrix.npy'), pairwise_dist)

    # save nodes
    print('Save nodes.txt...')
    filename = os.path.join(filepath, 'nodes.txt')
    with open(filename, 'w') as f:
        f.write("{}\t{}\n".format(len(G.nodes), 2))  # num_nodes, num_node_features
        for i, node in G.nodes.data():
            line = "\t".join(map(str, [i, node['x'], node['y']])) + "\n"
            f.write(line)

    # save edges
    print('Save edges.txt...')
    filename = os.path.join(filepath, 'edges.txt')
    with open(filename, 'w') as f:
        f.write("{}\t{}\n".format(len(G.edges), 2))
        for i, (fr, to, edge) in enumerate(G.edges.data()):
            maxspeed = edge['maxspeed'] if 'maxspeed' in edge else -1
            length = edge['length'] if 'length' in edge else -1
            if len(maxspeed) > 1:
                print('len(maxspeed) > 1')

            line = "\t".join(map(str, [i, fr, to, length, maxspeed])) + "\n"
            f.write(line)

    ox.plot_graph(G)

    location = [latMin / 2 + latMax / 2, longMin / 2 + longMax / 2]
    map_osm = Map(location=location,
                  min_lat=latMin,
                  max_lat=latMax,
                  min_lon=longMin,
                  max_lon=longMax,
                  zoom_start=11,
                  use_local_extrema=False, tiles="Stamen Terrain")
    upper_left = (latMax, longMin)
    upper_right = (latMax, longMax)
    lower_right = [latMin, longMax]
    lower_left = (latMin, longMin)

    bounds = [upper_left, upper_right, lower_right, lower_left]

    folium.Rectangle(bounds=bounds, color='#ff7800', fill=True, fill_color='#ffff00', fill_opacity=0.2).add_to(map_osm)
    folium.Marker([latMin, longMin], color='blue').add_to(map_osm)
    folium.Marker([latMax, longMax], color='red').add_to(map_osm)

    # map_osm.save("{timestamp}/viz_debug/{}.html".format(dataset_name))
    # webbrowser.open("{timestamp}/viz_debug/{}.html".format(dataset_name))


def get_ubodt(dataset_name, create_ubodt=False, delta=0.03):
    # Read network data
    network = Network(os.path.join("Datasets", dataset_name, "meta", 'graphInfo', "edges.shp"), "fid", "u", "v")
    print("Nodes {} edges {}".format(network.get_node_count(), network.get_edge_count()))
    graph = NetworkGraph(network)

    # Precompute an UBODT table. Can be skipped if you already generated an ubodt file
    if create_ubodt:
        ubodt_gen = UBODTGenAlgorithm(network, graph)
        status = ubodt_gen.generate_ubodt(os.path.join("Datasets", dataset_name, "meta",
                                                       'trajInfo', timestamp, "ubodt.txt"),
                                          delta=delta, binary=False, use_omp=True)
        print(status)

    # Read UBODT
    ubodt = UBODT.read_ubodt_csv(os.path.join("Datasets", dataset_name, "meta",
                                              'trajInfo', timestamp, "ubodt.txt"))

    return network, graph, ubodt


def get_nearest_points(point_coords, road_point_indexes,
                       node_strtree, k_nearest, radius_meters=200,
                       rbf_sigma=1e-4):
    """
    :param point_coords:
    :param road_point_indexes:
    :param node_strtree:
    :param k_nearest:
    :param radius_meters:
    :param rbf_sigma:
    :return: lat,long closest point to road from obs (N&K)
    """

    # Fiona uses (long, lat) format. Thus, we reverse.
    point_coords = Point(point_coords[::-1])
    """
    Want to consider only matches within max_dist meters. Than is, If gps points do not have a road point candidate 
    within the 200m is ignored. Thus, we need to transform decimal degrees into meters. According to 
    200 * 360 / (2 * np.pi * 6371000), a distance of 200m corresponds to 0.0017986432118374611 
    decimal degrees. Therefore, 200m is approximately 0.002 decimal degrees.
    """
    distance = radius_meters * 360 / (2 * np.pi * 6371000)
    nrst_points = [(road_point_indexes[id(pt)], pt) for pt in node_strtree.query(point_coords.buffer(distance))]
    if len(nrst_points) == 0:
        print("No matching roads found!")
        return None
    dist = [point_coords.distance(cp) for _, cp in nrst_points]
    idx = np.argsort(dist)  # ascending order

    nrst_points = [nrst_points[i] for i in idx]
    dist = [dist[i] for i in idx]

    def softmax(smallest_distances):
        """Compute softmax values for each sets of scores in x."""
        # RBF and re normalization
        weights_neighbors = np.exp(-np.array(smallest_distances) / rbf_sigma)
        weights_neighbors /= (weights_neighbors.sum(axis=0) + 1e-8)
        return weights_neighbors

    nrst_points = [(*cp, d) for cp, d in zip(nrst_points, softmax(dist))]

    # Convert back to (lat, long) from (long, lat)
    nrst_points = [(_id, transform(lambda x, y: (y, x), _cp), _dst) for _id, _cp, _dst in nrst_points]
    if len(nrst_points) < k_nearest:
        assert ValueError("No matching neighbours in range.")

    return nrst_points[:k_nearest]


def plot_traj(gps_traj, df_all, match_result,
              dataset_name, latMin, latMax, longMin, longMax,
              plot_marks=False,
              plot_mapped_path=False,
              plot_density=True):
    array_export_wkt = np.array(wkt.loads(match_result.pgeom.export_wkt()).coords)

    location = [latMin / 2 + latMax / 2, longMin / 2 + longMax / 2]
    map_osm = Map(location=location,
                  min_lat=latMin,
                  max_lat=latMax,
                  min_lon=longMin,
                  max_lon=longMax,
                  zoom_start=11,
                  use_local_extrema=False,
                  tiles="Stamen Terrain")
    # list of points (latitude, longitude)) – Latitude and Longitude of line (Northing, Easting)
    # points = [(latMin, longMin), (latMax, longMax)]
    # Define dimensions of box in grid
    upper_left = (latMax, longMin)
    lower_right = [longMax, latMin]
    # Define json coordinates for polygon
    bounds = [upper_left, lower_right]
    folium.Rectangle(bounds=bounds, color='#ff7800', fill=True, fill_color='#ffff00', fill_opacity=0.2).add_to(map_osm)

    if plot_marks:
        # add markers to map
        for _, row in gps_traj.iterrows():
            lat, lon = row[["lat", "lon"]]

            folium.CircleMarker((lat, lon), radius=3, color='blue', fill=True,
                                fill_color='#3186cc', fill_opacity=0.5,
                                parse_html=False).add_to(map_osm)
        for row in array_export_wkt:
            lat, lon = row[1], row[0]

            folium.CircleMarker((lat, lon), radius=3, color='green', fill=True,
                                fill_color='#3186cc', fill_opacity=0.5,
                                parse_html=False).add_to(map_osm)

    # Enable to plot the mapped path
    if plot_mapped_path:
        mr_wkt = match_result.mgeom.export_wkt()
        traj_style = {
            'fillColor': 'green',
            'color': 'green',
            'opacity': 0.8,
            'weight': 3
        }
        layer = folium.GeoJson(
            data=mapping(loads(mr_wkt)),
            name='geojson',
            style_function=lambda x: traj_style
        ).add_to(map_osm)  # 1. keep a reference to GeoJSON layer

        map_osm.fit_bounds(layer.get_bounds())  # 2. fit the map to GeoJSON layer

    if plot_density:
        stationArr = df_all[["lat", "lon"]].to_numpy()  # plot heatmap
        map_osm.add_child(plugins.HeatMap(stationArr, radius=5))

    # map_osm.save("{timestamp}/viz_debug/{}.html".format(dataset_name))
    # webbrowser.open("{timestamp}/viz_debug/{}.html".format(dataset_name))


def save_weights(traj_gb, dataset_name, save, plt_weights, rbf_sigma):
    # Map each observation to the k_closest OSM nodes
    print('Save observations.txt')

    # Read in roads shapefiles using fiona. Fiona uses (long, lat) format !!!!!
    roads_point_shp = fiona.open(os.path.join("Datasets", dataset_name, "meta", 'graphInfo', "nodes.shp"))
    print('roads_shp.crs: {}'.format(roads_point_shp.crs))  # roads_shp.crs: {'init': 'epsg:4326'}
    print('roads_shp.schema: {}'.format(roads_point_shp.schema))

    # Note: The data in Open Street Map database is stored in a gcs with units decimal
    # degrees & datum of wgs84. (EPSG: 4326)
    # Just preserving geometries for my purpose. Data are in (long, lat) format.
    road_point_geoms = [shape(shp['geometry']) for shp in roads_point_shp]

    n_nodes = len(road_point_geoms)
    nx_nodes = np.zeros([n_nodes, 2])
    for i, d in enumerate(road_point_geoms):
        nx_nodes[i, 0] = d.y  # latitude
        nx_nodes[i, 1] = d.x  # longitude

    # Create STR-tree as input to particle filter
    node_strtree = STRtree(road_point_geoms)

    # To get the original indexes of the query results, create an auxiliary dictionary.
    # But use the geometry ids as keys since the shapely geometries themselves are not hashable.
    road_point_indexes = dict((id(pt), i) for i, pt in enumerate(road_point_geoms))

    k_nearest = 10

    all_weights_neighbors = []
    all_closest_node_index = []

    # gps_obs = pd.concat(traj_gb).reset_index(drop=True)
    for gps_traj in tqdm(traj_gb, desc='Compute observations...'):
        per_traj_weights_neighbors = []
        per_traj_closest_node_index = []
        per_traj_nearest_points = []

        traj_id = set(gps_traj['traj_id']).pop()
        for _, gps_item in gps_traj.iterrows():
            # target GPS
            point_coords = gps_item[['lat', 'lon']]

            # # point_coords are reversed inside get_nearest_points
            _nearest_points = get_nearest_points(point_coords=point_coords,
                                                 road_point_indexes=road_point_indexes,
                                                 node_strtree=node_strtree,
                                                 k_nearest=k_nearest,
                                                 radius_meters=1250,
                                                 rbf_sigma=rbf_sigma)

            if _nearest_points or len(_nearest_points) < k_nearest:
                per_traj_nearest_points.append(np.array([(n_p[1].x, n_p[1].y) for n_p in _nearest_points]))
                per_traj_weights_neighbors.append(np.array([wn[2] for wn in _nearest_points]))
                per_traj_closest_node_index.append(np.array([wn[0] for wn in _nearest_points]))
            else:
                warnings.warn("No points are found within range")

        all_weights_neighbors.extend(per_traj_weights_neighbors)
        all_closest_node_index.extend(per_traj_closest_node_index)

        tr_weights_neighbors = np.vstack(per_traj_weights_neighbors).T
        tr_closest_node_index = np.vstack(per_traj_closest_node_index).T

        # Save and plot weights for each trajectory
        location = nx_nodes.mean(axis=0)
        map_ = Map(location=location,
                   zoom_start=11,
                   use_local_extrema=False, tiles="Stamen Terrain")

        # point_idx = [8, 30, 100, 200, 300, 400]
        point_idx = list(range(gps_traj.shape[0]))
        for pt_idx in point_idx:
            # GPS points
            lat, lng = gps_traj.iloc[pt_idx]['lat'], gps_traj.iloc[pt_idx]['lon']
            popup = "observation with id {}".format(pt_idx)
            folium.CircleMarker(location=(lat, lng), popup=popup, radius=5).add_to(map_)

        latMin = min(gps_traj.lat)
        latMax = max(gps_traj.lat)
        longMin = min(gps_traj.lon)
        longMax = max(gps_traj.lon)
        nx_nodes_box_flags = (nx_nodes[:, 1] > longMin) & \
                             (nx_nodes[:, 1] < longMax) & \
                             (nx_nodes[:, 0] > latMin) & \
                             (nx_nodes[:, 0] < latMax)
        nx_nodes_box = nx_nodes[nx_nodes_box_flags]
        # Plot graph (map) nodes
        for nx_nodes_box_point in nx_nodes_box:
            lat, lng = nx_nodes_box_point
            folium.CircleMarker(location=(lat, lng), color='green',
                                fill_color='green',
                                line_color=None,
                                fill_opacity=0.8, radius=5).add_to(map_)

        # mapping to k closest
        for pt_idx in point_idx:
            for i, closest_idx in enumerate(tr_closest_node_index[:, pt_idx]):
                lat, lng = nx_nodes[closest_idx]
                weight = tr_weights_neighbors[i, pt_idx]
                popup = "closest to observation id {}, weight {:.2f}".format(pt_idx, weight)
                folium.CircleMarker(location=(lat, lng), popup=popup, radius=50 * weight, color='red').add_to(map_)
        Path(f"viz_debug/{dataset_name}/{timestamp}/weight_debug").mkdir(parents=True, exist_ok=True)
        map_.save(f"viz_debug/{dataset_name}/{timestamp}/weight_debug/{traj_id}.html")
        if plt_weights:
            webbrowser.open(f"viz_debug/{dataset_name}/{timestamp}/weight_debug/tr_{traj_id}.html")

    all_weights_neighbors = np.vstack(all_weights_neighbors).T
    all_closest_node_index = np.vstack(all_closest_node_index).T

    """
     Observations file start with num_observations, k (point per observation) then per line node_id, weight x k. 
     Observations are the GPS points.
     Example:
        ```
        2518	5
        17025	0.22376753215971462	17026	0.2186635904321353	1137	0.18742442008753432	6888	0.20024607632540276	4585	0.16989838099521318
        6888	0.20106576291692577	1137	0.20348475328200213	4585	0.20255400616332436	1139	0.1985437138699239	6887	0.1943517637678238
        14928	0.18319982750248237	1302	0.18136407620166017	14929	0.1979849150163569	628	0.18905104643181994	1303	0.24840013484768056
        ```
    """
    if save:
        # save observations
        filename = os.path.join("Datasets", dataset_name, "meta", 'trajInfo', timestamp, 'observations.txt')

        with open(filename, 'w') as f:
            f.write("{}\t{}\n".format(all_weights_neighbors.shape[1], all_weights_neighbors.shape[0]))  # num_obs, k
            for j in range(all_weights_neighbors.shape[1]):
                ids = all_closest_node_index[:, j]
                weights = all_weights_neighbors[:, j]
                obs = "\t".join(["{}\t{}".format(i, w) for i, w in zip(ids, weights)])
                f.write(obs + "\n")


def save_lengths(traj_gb, dataset_name):
    print('save lengths.txt')

    filepath = os.path.join('Datasets', dataset_name, 'meta', 'trajInfo', timestamp)
    filename = os.path.join(filepath, 'lengths.txt')
    with open(filename, 'w') as f:
        for gps_traj in tqdm(traj_gb):
            length = len(gps_traj)
            traj_id = set(gps_traj['traj_id']).pop()
            line = "{}\t{}\n".format(traj_id, length)
            f.write(line)


# rgb tuple to hexadecimal conversion
def rgb2hex(rgb):
    rgb = [hex(int(256 * x)) for x in rgb]
    r, g, b = [str(x)[2:] for x in rgb]
    return f"#{r}{g}{b}"


class Leg(object):

    def __init__(self, start_node, target_node, traversed_edges, steps):
        self.start_node = start_node
        self.target_node = target_node
        self.traversed_edges = traversed_edges
        self.steps = steps


def remove_consecutive_duplicates(inseq):
    if not inseq:
        return inseq
    return [inseq[0]] + [b for a, b in zip(inseq, inseq[1:]) if a != b]


class NoConfidentPathFound(Exception):
    pass


def nodes_to_edges(nodes):
    return [(fr, to) for fr, to in zip(nodes, nodes[1:])]  # if (fr, to, 0) in lausanne.edges]  # should remove this


class JsonError(Exception):
    pass


def find_edge_id_from_observations(observations, graph_map, org_graph_map):
    """
    Use HMM to map observations to edges on the map
    return a list of (from_node, to_node, list_edges) for each leg
    """
    try:
        coords_str = ";".join("{},{}".format(lon, lat) for (lon, lat) in observations)
        # doc: https://github.com/Project-OSRM/osrm-backend/blob/master/docs/http.md
        query = 'https://router.project-osrm.org/match/v1/car/{}?steps=false&annotations=true&tidy=true'.format(
            coords_str)
        result = requests.get(query)
        from json.decoder import JSONDecodeError
        try:
            data = json.loads(result.text)
        except JSONDecodeError as e:
            raise JsonError('Json error')

        legs = data['matchings'][0]['legs']
        if len(legs) != len(observations) - 1:
            raise (WrongNumberOfLegs(
                "{} observations results in {} legs instead of {}".format(len(observations), len(legs),
                                                                          len(observations) - 1)))
            # return None, None
        if data['matchings'][0]['confidence'] < 0.1:
            raise (NoConfidentPathFound("confidence {} is too low".format(data['matchings'][0]['confidence'])))
            # return None, None

        # mapped_nodes_id = dict((osm, i) for i, osm in graph_map.nodes(data=True))
        mapped_nodes_id = dict((osm_id, (i, data)) for i, (osm_id, data) in enumerate(org_graph_map.nodes(data=True)))
        mapped_edges_id = dict(((fr, to), i) for i, (fr, to, osm) in enumerate(graph_map.edges.data('osmid')))
        results = []

        legs_points_xy = []
        for leg in legs:
            nodes = leg['annotation']['nodes']
            nodes_ids = [mapped_nodes_id[n][0] for n in nodes]  # if n in mapped_nodes_id]
            nodes_xy = [(mapped_nodes_id[n][1]['y'], mapped_nodes_id[n][1]['x']) for n in nodes]
            legs_points_xy.append(nodes_xy)
            # nodes_ids = [n for n in nodes if n in mapped_nodes_id]
            if not nodes_ids:
                results.append(None)
                continue

            edges = nodes_to_edges(nodes_ids)  # find edges (fr, to)
            edges_id = remove_consecutive_duplicates([mapped_edges_id[edge] for edge in edges])
            try:
                num_steps = nx.shortest_path_length(graph_map, nodes_ids[0], nodes_ids[-1])
            except nx.NetworkXNoPath:
                print("\nno path")
                num_steps = 20

            results.append(
                Leg(start_node=nodes_ids[0], target_node=nodes_ids[-1], traversed_edges=edges_id, steps=num_steps))
    except (WrongNumberOfLegs, NoConfidentPathFound, KeyError) as e:
        return None, None

    return results, legs_points_xy


def read_from_files(nodes_filename: str, edges_filename: str):
    """
    Load a graph from files `nodes.txt` and 'edges.txt`

    Node file starts with number of nodes, number of features per node
    Followed by one line per node, id then features. Example:
    ```
    18156	2
    0	6.6491811	46.5366765
    1	6.6563029	46.5291637
    2	6.6488104	46.5365551
    3	6.6489423	46.5367163
    4	6.649007	46.5366124
    5	6.5954845	46.5224695
    ...
    ```

    Edge file starts with number of edges, number of features per edges
    Followed by one line per edge: id, from_node, to_node, then features. Example:

    ```
    32468	2
    0	0	6	11.495	50
    1	1	10517	23.887	20
    2	1	10242	8.34	20
    3	2	4	16.332	50
    4	2	11342	13.31	-1
    5	2	6439	15.761	50
    6	2	11344	15.797	50
    ```
    """
    node_features = None
    edge_features = None

    # read node features
    with open(nodes_filename) as f:
        num_nodes, num_node_features = map(int, f.readline().split('\t'))
        if num_node_features > 0:
            node_features = torch.zeros(num_nodes, num_node_features)
            for i, line in enumerate(f.readlines()):
                features = torch.tensor(
                    list(map(float,
                             line.split('\t')[1:])))
                node_features[i] = features

    # read edge features
    with open(edges_filename) as f:
        num_edges, num_edge_features = map(int, f.readline().split('\t'))

        senders = torch.zeros(num_edges, dtype=torch.long)
        receivers = torch.zeros(num_edges, dtype=torch.long)

        if num_edge_features > 0:
            edge_features = torch.zeros(num_edges, num_edge_features)

        for i, line in enumerate(f.readlines()):
            elements = line.split('\t')
            senders[i] = int(elements[1])
            receivers[i] = int(elements[2])
            if edge_features is not None:
                edge_features[i] = torch.tensor(
                    list(map(float, elements[3:])))

    return edge_features, senders, receivers


def mapped_paths(G, org_G, traj_gb, dataset_name,
                 delta=0.03, FastMap=False,
                 create_ubodt=False, save=False,
                 plt_paths=True, verbose=False):
    # Compute mapped_edges_id and mapped_nodes_id mostly for verifications below.
    mapped_edges_id = dict(((fr, to), (i, data)) for i, (fr, to, data) in enumerate(G.edges(data=True)))
    mapped_nodes_id = dict((i, (osm_id, data)) for i, (osm_id, data) in enumerate(G.nodes(data=True)))

    print('Compute mapped paths.txt')
    mapped_traj_gb = []
    if FastMap:
        # Slower but more accurate map matching (FMM), https://github.com/cyang-kth/fmm
        network, graph, ubodt = get_ubodt(dataset_name, create_ubodt=create_ubodt, delta=delta)

        # Create FMM model
        model = FastMapMatch(network, graph, ubodt)
        # Define map matching configurations
        # How to select candidate size k, search radius r and GPS error e?
        # k: 16 to 32
        # r: 300 meters or above (if you are using GPS data in unit of degrees, setting the parameter as 0.003 degrees)
        # e: 50 to 100 meters (if you are using GPS data in unit of degrees, setting the parameter as 0.005 degrees)

        # "k": 32,
        # "r": 0.01,
        # "e": 0.003
        _config = FastMapMatchConfig(k_arg=16,
                                     r_arg=300 / 1.132e5,
                                     gps_error=500 / 1.132e5,
                                     reverse_tolerance=0)
    else:
        # Faster but more accurate map matching algorithm (STMatch), https://github.com/cyang-kth/fmm
        # Read network data
        print('Read network data for stmatch...')
        network = Network(os.path.join("Datasets", dataset_name, "meta", 'graphInfo', "edges.shp"), "fid", "u", "v")
        print("Nodes {} edges {}".format(network.get_node_count(), network.get_edge_count()))
        graph = NetworkGraph(network)
        model = STMATCH(network, graph)
        _config = STMATCHConfig(k_arg=32,
                                r_arg=300 / 1.132e5,
                                gps_error_arg=150 / 1.132e5,
                                vmax_arg=30 / 3.6 / 1.132e5,
                                factor_arg=1.5)

    failed_files = []
    hmm_edges = []
    # for idx, (_, gps_traj) in tqdm(traj_gb.iterrows(), total=traj_gb.shape[0]):
    for idx, gps_traj in enumerate(tqdm(traj_gb)):

        traj_id = set(gps_traj['traj_id']).pop()
        print('Process traj_id {}'.format(traj_id))
        # Interpolate signal !
        if dataset_name == 'Geolife':
            coords = gps_traj[['lon', 'lat', 'alt', 'time']]
        else:
            coords = gps_traj[['lon', 'lat']]
        # coords = interpolate_gps(coords, method=2)

        # Run map matching for wkt
        points = GeoSeries(map(Point, zip(coords['lon'], coords['lat'])))
        input_wkt = LineString(points.tolist())

        if FastMap:
            match_result = model.match_wkt(input_wkt.wkt, _config)
            if not len(list(match_result.cpath)):  # == len(points):
                print("Failed on idx: {} traj_id: {}".format(idx, traj_id))
                failedItem = {'idx': idx, 'traj_id': traj_id}
                failed_files.append(failedItem)
                continue
        else:
            match_result = model.match_wkt(input_wkt.wkt, _config)

            if not len(list(match_result.cpath)):  # == len(points):
                continue
        if verbose:
            # Print map matching match_result
            print("Matched path: ", list(match_result.cpath))  # traversed edges
            print("Matched edge for each point: ", list(match_result.opath))
            print("Matched edge index ", list(match_result.indices))
            print("Matched geometry: ", match_result.mgeom.export_wkt())

        # edge_features, senders, receivers = \
        #     read_from_files(os.path.join("Datasets", dataset_name, "meta", "nodes.txt"),
        #                     os.path.join("Datasets", dataset_name, "meta", "edges.txt"))

        # mgeom_xy = wkt.loads(match_result.mgeom.export_wkt()).xy
        opath = list(match_result.opath)  # edge matched to each point in trajectory (edge ids (osmid ?))
        opath_indexes = [network.get_edge_index(_edge_id) for _edge_id in opath]

        cpath = list(match_result.cpath)  # the path traversed by the trajectory (edge ids)
        cpath_index_dict = {network.get_edge_index(_edge_id): i for i, _edge_id in enumerate(cpath)}
        cpath_indexes = [network.get_edge_index(_edge_id) for _edge_id in cpath]

        legs = []
        legs_points_xy = []
        # For each leg compute a list of (from_node, to_node, list_edges)
        for opath_idx_fr, opath_idx_to in zip(opath_indexes, opath_indexes[1:]):
            cpath_i_fr, cpath_i_to = cpath_index_dict[opath_idx_fr], cpath_index_dict[opath_idx_to]
            if cpath_i_to >= cpath_i_fr:
                intermediate_edge_indexes = cpath_indexes[cpath_i_fr:cpath_i_to + 1]
            else:
                intermediate_edge_indexes = cpath_indexes[cpath_i_to:cpath_i_fr + 1]

            # # ---------> Here STARTS part of the code for visualization purposes
            # leg_points_xy contains the graph node points between two consecutive GPS points
            leg_points_xy = []
            temp = []
            mapped_leg_point_indexes = []
            for _edge_idx in intermediate_edge_indexes:  # Edge indexes between two GPS points
                # node_fr_idx = network.get_edge(_edge_idx).source
                # get the target node of the first edge
                edge_id = network.get_edge_id(_edge_idx)
                assert _edge_idx == edge_id, '_edge_idx == edge_id is False'
                # node_to_idx = network.get_edge(edge_id).target

                fr = network.get_node_id(network.get_edge(edge_id).source)  # id of the re-labeled graph
                to = network.get_node_id(network.get_edge(edge_id).target)
                temp.append(mapped_edges_id[(fr, to)][0])

                mapped_leg_point_indexes.append(fr)
                mapped_leg_point_indexes.append(to)
                # G.get_edge_data(fr, to)

                assert G.get_edge_data(fr, to), 'No edge found'

                # Just to validate node coords
                _fr_to_points = [(mapped_nodes_id[fr][1]['y'], mapped_nodes_id[fr][1]['x']),
                                 (mapped_nodes_id[to][1]['y'], mapped_nodes_id[to][1]['x'])]

                fr_to_points = [(p_lat, p_lon) for p_lon, p_lat in
                                zip(*loads(network.get_edge_geom(edge_id).export_wkt()).xy)]

                # assert all([(np.subtract(item1, item2) < 1e-5).all()
                #             for item1, item2 in zip(_fr_to_points, fr_to_points)]), 'node coords not match'

                leg_points_xy.extend([(p_lat, p_lon) for p_lon, p_lat in
                                      zip(*loads(network.get_edge_geom(edge_id).export_wkt()).xy)][:-1])
            legs_points_xy.append(leg_points_xy)
            # # ---------> Here ENDS part of the code for visualization purposes

            edges_id = remove_consecutive_duplicates(intermediate_edge_indexes)

            legs.append(Leg(start_node=mapped_leg_point_indexes[0], target_node=mapped_leg_point_indexes[-1],
                            traversed_edges=edges_id, steps=5))

        # observations = list(zip(coords.lon, coords.lat))
        # _legs, _legs_points_xy = find_edge_id_from_observations(observations, G, org_G)

        if legs:
            # # keep successfully mapped trajectories
            mapped_traj_gb.append(gps_traj)
            hmm_edges.append(legs)
        else:
            continue

        if len(legs) != len(coords) - 1:
            raise (WrongNumberOfLegs("{} observations (GPS) results in {} legs instead of {}"
                                     .format(len(coords), len(legs), len(coords) - 1)))

        # You can use this code to visualize the mapped paths along with the graph nodes between GPS points
        color_mapper = cm.ScalarMappable(cmap=cm.cividis)
        rgb_values = [c[:3] for c in
                      color_mapper.to_rgba(np.arange(len(legs_points_xy)))]  # keep rgb and drop the "a" column
        # colors = [rgb2hex(rgb) for rgb in rgb_values]
        fill_colors = cycle(['yellow', 'black'])
        if dataset_name == 'Geolife':
            town_coord = [39.904202, 116.407394]
        else:
            town_coord = [40.636228, 22.947501]

        map_ = Map(town_coord,
                   zoom_start=14,
                   tiles="Stamen Terrain")
        cnt = 0
        for i, edge_leg_points in enumerate(legs_points_xy):
            folium.PolyLine(edge_leg_points, color='red', weight=4.5, opacity=1).add_to(map_)
            fill_color = next(fill_colors)
            for edge_leg_point in edge_leg_points:
                folium.Circle(edge_leg_point,
                              radius=5.,
                              popup='{} : '.format(cnt) + ', '.join(map(str, edge_leg_point)),
                              tooltip='{} : '.format(cnt) + ', '.join(map(str, edge_leg_point)),
                              fill_color=fill_color,
                              line_color="black",
                              fill_opacity=0.8,
                              fill=True
                              ).add_to(map_)
                cnt += 1
        # Plot projected (matched) points in Blue
        for lon, lat in np.vstack(wkt.loads(match_result.pgeom.export_wkt()).xy).T[:len(legs_points_xy) + 1]:
            folium.Marker([lat, lon]).add_to(map_)
        # Plot raw points in Red
        for _, raw_point in coords.iterrows():
            folium.Marker([raw_point.lat, raw_point.lon], icon=folium.Icon(color='red')).add_to(map_)
        # Save and reopen html path.
        Path(f"viz_debug/{dataset_name}/{timestamp}/mapped").mkdir(parents=True, exist_ok=True)
        map_.save(f"viz_debug/{dataset_name}/{timestamp}/mapped/{traj_id}.html")
        if plt_paths:
            webbrowser.open(f"viz_debug/{dataset_name}/{timestamp}/mapped/tr_{traj_id}.html")

    if save:

        """
         Paths file start with number of paths and maximum length
            Then per line, sequence of traversed edge ids. Example:
            ```
            2254	41
            20343	30411	30413	12311	1946
            1946	8179	30415	24401	24403	1957	8739	1960	24398	24400	20824	20822	20814	19664	19326	19327	26592	19346	29732	26594	13778	20817	13785	26595	26597
            ```
        """
        import json
        print("Number of failed_files: {}".format(len(failed_files)))
        filepath = os.path.join('Datasets', dataset_name, 'meta', 'trajInfo', timestamp)

        Path(filepath).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(filepath, 'failed_files.txt')
        with open(filename, 'w') as f:
            json.dump(failed_files, f, indent=4)

        assert len(mapped_traj_gb), 'No successfully mapped trajectories'

        df = pd.concat(mapped_traj_gb)
        n_legs = sum(len(legs) for legs in hmm_edges)
        n_traj = len(df.traj_id.unique())
        lengths = df.traj_id.value_counts().sort_index()
        assert lengths.sum() == len(df)

        assert len(hmm_edges) == len(df.traj_id.unique())

        assert len(lengths) == n_traj
        assert n_legs == len(df) - n_traj  # -n_traj the number of legs per traj are len(traj)-1

        # save paths by leg
        filename = os.path.join(filepath, 'paths.txt')
        with open(filename, 'w') as f:
            all_traversed_edges = [leg.traversed_edges for legs in hmm_edges for leg in legs]
            # all_traversed_edges = [leg for legs in hmm_edges for leg in legs]
            longest_path_length = max(map(len, all_traversed_edges))
            f.write("{}\t{}\n".format(len(all_traversed_edges), longest_path_length))
            for traversed_edges in all_traversed_edges:
                line = "\t".join(map(str, traversed_edges)) + "\n"
                f.write(line)

    return mapped_traj_gb


def plot_mapped(match_result, map_osm):
    mr_wkt = match_result.mgeom.export_wkt()
    traj_style = {
        'fillColor': 'green',
        'color': 'green',
        'opacity': 0.8,
        'weight': 3
    }
    layer = folium.GeoJson(
        data=mapping(loads(mr_wkt)),
        name='geojson',
        style_function=lambda x: traj_style
    ).add_to(map_osm)  # 1. keep a reference to GeoJSON layer

    map_osm.fit_bounds(layer.get_bounds())  # 2. fit the map to GeoJSON layer

    return map_osm


# USGS Elevation Point Query Service
url = r'https://nationalmap.gov/epqs/pqs.php?'


def elevation_function(df, lat_column, lon_column):
    """Query service using lat, lon. add the elevation values as a new column."""
    elevations = []
    for lat, lon in zip(df[lat_column], df[lon_column]):
        # define rest query params
        params = {
            'output': 'json',
            'x': lon,
            'y': lat,
            'units': 'Meters'
        }

        # format query string and return query value
        result = requests.get((url + urllib.parse.urlencode(params)))
        elevations.append(result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])

    df['elev_meters'] = elevations


def interpolate_gps(coords, method=2):
    coords = coords.sort_values(by="time").reset_index(drop=True)
    # coords = coords[coords['time'].ne(coords['time'].shift())]
    coords.set_index('time', inplace=True, drop=True)
    coords = coords[~coords.index.duplicated()]

    coords['idx'] = range(0, len(coords))

    if method == 1:
        # import pyproj
        # proj = pyproj.Transformer.from_crs(3857, 4479)
        # lat, lon = proj.transform(coords.lat, coords.lon)
        # coords.lat = lat
        # coords.lon = lon

        print(coords.head())
        coords.index = np.round(coords.index.astype(np.int64), -9).astype('datetime64[ns]')

        coords = coords.resample('1S').asfreq()

        measurements = np.ma.masked_invalid(coords[['lon', 'lat', 'alt']].values)

    else:
        import datetime
        METHOD_MAX_GAP = 2000  # seconds
        """
        Method 2: fill the gaps between points close to each other's with NaNs and leave the big holes alone.
        The resulting sampling time is not constant
        """
        for i in range(0, len(coords) - 1):
            gap = coords.index[i + 1] - coords.index[i]
            if datetime.timedelta(seconds=1) < gap <= datetime.timedelta(seconds=METHOD_MAX_GAP):
                gap_idx = pd.date_range(start=coords.index[i] + datetime.timedelta(seconds=1),
                                        end=coords.index[i + 1] - datetime.timedelta(seconds=1),
                                        freq='1S')
                gap_coords = pd.DataFrame(coords, index=gap_idx)
                coords = coords.append(gap_coords)
                # print("Added {} points in between {} and {}"
                # .format(len(gap_idx), coords.index[i], coords.index[i + 1]))
        # Sort all points
        coords = coords.sort_index()
        # Fill the time_sec column
        coords['time_sec'] = np.nan
        for i in range(0, len(coords)):
            coords['time_sec'].values[i] = (coords.index[i] - datetime.datetime(2000, 1, 1, 0, 0, 0)).total_seconds()
        coords['time_sec'] = coords['time_sec'] - coords['time_sec'][0]
        # Create the "measurement" array and mask NaNs
        measurements = coords[['lon', 'lat', 'alt']].values
        measurements = np.ma.masked_invalid(measurements)

    # Covariances: Position = 0.0001deg = 11.1m, Altitude = 30m
    cov = {'coordinates': 1.,
           'elevation': 30.,
           'horizontal_velocity': 1e-3,
           'elevation_velocity': 1e-3,
           'horizontal_acceleration': 1e-6 * 1000,
           'elevation_acceleration': 1e-6 * 1000}
    if method == 1:
        transition_matrices = np.array([[1, 0, 0, 1, 0, 0],
                                        [0, 1, 0, 0, 1, 0],
                                        [0, 0, 1, 0, 0, 1],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]])
    else:
        # The samples are randomly spaced in time, so dt varies with time and a
        # time dependent transition matrix is necessary
        timesteps = np.asarray(coords['time_sec'][1:]) - np.asarray(coords['time_sec'][0:-1])
        transition_matrices = np.zeros(shape=(len(timesteps), 6, 6))
        for i in range(len(timesteps)):
            transition_matrices[i] = np.array([[1., 0., 0., timesteps[i], 0., 0.],
                                               [0., 1., 0., 0., timesteps[i], 0.],
                                               [0., 0., 1., 0., 0., timesteps[i]],
                                               [0., 0., 0., 1., 0., 0.],
                                               [0., 0., 0., 0., 1., 0.],
                                               [0., 0., 0., 0., 0., 1.]])
    observation_matrices = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0]])

    # observation_covariance = np.diag([1e-4, 1e-4, 100]) ** 2
    observation_covariance = np.diag([cov['coordinates'], cov['coordinates'], cov['elevation']]) ** 2

    initial_state_mean = np.hstack([measurements[0, :], 3 * [0.]])
    # works initial_state_covariance = np.diag([1e-3, 1e-3, 100, 1e-4, 1e-4, 1e-4])**2
    # initial_state_covariance = np.diag([1e-4, 1e-4, 50, 1e-6, 1e-6, 1e-6]) ** 2
    initial_state_covariance = np.diag([cov['coordinates'], cov['coordinates'], cov['elevation'],
                                        cov['horizontal_velocity'], cov['horizontal_velocity'],
                                        cov['elevation_velocity']]) ** 2

    kf = KalmanFilter(transition_matrices=transition_matrices,
                      observation_matrices=observation_matrices,
                      observation_covariance=observation_covariance,
                      initial_state_mean=initial_state_mean,
                      initial_state_covariance=initial_state_covariance,
                      # em_vars=['transition_matrices',
                      #          'observation_matrices',
                      #          'transition_offsets',
                      #          'observation_offsets',
                      #          'transition_covariance',
                      #          'observation_covariance',
                      #          'initial_state_mean',
                      #          'initial_state_covariance'
                      #          ]
                      )

    kf = kf.em(measurements, n_iter=5)
    state_means, state_vars = kf.smooth(measurements)
    # Increase observation co-variance
    kf.observation_covariance = kf.observation_covariance * 10

    filled_coords = state_means[coords['idx'].isnull(), :3]
    orig_coords = coords[~coords['idx'].isnull()].set_index('idx')

    # fig = plt.figure()
    #
    plt.plot(orig_coords['lon'], orig_coords['lat'], 'r.', markersize=2, label='original')
    # plt.plot(filled_coords[:, 0], filled_coords[:, 1], 'go', markersize=2, label='filled')
    plt.legend()
    # plt.savefig('output.png')

    plt.show()

    # a = coords.copy(deep=True)
    coords.loc[coords['idx'].isnull(), 'lon'] = state_means[coords['idx'].isnull(), 0]
    coords.loc[coords['idx'].isnull(), 'lat'] = state_means[coords['idx'].isnull(), 1]
    coords.loc[coords['idx'].isnull(), 'alt'] = state_means[coords['idx'].isnull(), 2]

    # return pd.DataFrame(filled_coords, columns=['lon', 'lat', 'alt'])
    return coords


def show_trace(df, traj_id, filename, dataset_name, plt_path=False, save=False):
    if not save:
        return
    if plt_path and not save:
        return
    if df.empty:
        return

    traj_df = df[df.traj_id == traj_id]
    map_ = Map([traj_df.lat.mean(), traj_df.lon.mean()], zoom_start=11, tiles="Stamen Terrain")
    for i, row in traj_df.iterrows():
        if dataset_name == 'Geolife':
            popup = "traj_id {} time {}".format(row.traj_id, row.time)
        else:
            popup = "traj_id {}".format(row.traj_id)
        folium.Marker([row.lat, row.lon], popup=popup).add_to(map_)

    if save:
        map_.save(filename)
    if plt_path:
        webbrowser.open(filename)


def rm_time_close(df, min_time_delta):
    if df.empty:
        return df
    # # 1. Remove observation that are less than min_time_delta sec away
    min_delta = np.timedelta64(min_time_delta, 's')
    repetition_points = (pd.to_datetime(df.time).diff() < min_delta) & (df.traj_id.diff() == 0)
    print('remove {:.1f}% of points because they are less than {}s from previous obs'.format(
        repetition_points.mean() * 100, min_time_delta))
    df = df.loc[~repetition_points]
    df = df.reset_index(drop=True)
    return df


def rm_dist_close(df, min_distance, max_distance, disable_tqdm=True):
    if df.empty:
        return df

    df = df.reset_index()

    # # if False:
    coords = list(zip(df.lat, df.lon))
    distances_with_previous = np.array([0] + [geodesic(p1, p2).m
                                              for p1, p2 in tzip(coords, coords[1:], disable=disable_tqdm)])

    keep = ((min_distance < distances_with_previous) & (
            distances_with_previous < max_distance))

    print("remove {}% of points because they are less than {} meters from previous obs"
          .format(100. * (1 - keep.mean()), min_distance))

    df = df.loc[keep]
    df = df.reset_index(drop=True)
    df['traj_id'] = df['traj_id'].astype(int)

    return df


def rm_short_traj(df, max_len=100):
    _traj_gb = list(df.groupby('traj_id'))
    traj_gb = []
    for traj in _traj_gb:
        if len(traj) > max_len:
            traj_gb.append(traj)

    df = pd.concat(traj_gb)

    return df


def rm_in_diameter(df, path_min_diameter=50):
    if df.empty or len(df) < 2:
        return df

    n_obs = len(df)
    observations = np.zeros([n_obs, 2])
    observations[:, 0] = df.lat
    observations[:, 1] = df.lon

    lengths = df.traj_id.value_counts().sort_index()
    start_index = np.zeros(len(lengths)).astype(int)
    start_index[1:] = lengths.cumsum()[:-1]

    diameter = lengths.copy()
    for i, (traj_id, start, length) in enumerate(zip(lengths.index, start_index, lengths)):
        points = observations[start:start + length, :]
        # dists = pdist(points)
        sc_dist = cdist(points, points, lambda u, v: geodist(u, v).meters)  # you can choose unit here

        diameter.loc[traj_id] = sc_dist.max()

    # from scipy.spatial.distance import cdist
    # from geopy.distance import distance as geodist  # avoid naming confusion# %%
    # # if False:
    # sc_dist = cdist(observations, observations, lambda u, v: pdist(u, v))  # you can choose unit here

    orders_too_concentrated = diameter[diameter < path_min_diameter].sort_values().index
    # show longest to remove for illustration
    # show_trace(df, orders_too_concentrated[-1])

    # %%
    # if False:

    orders_too_concentrated = orders_too_concentrated.values
    keep = ~df.traj_id.isin(orders_too_concentrated)
    print('remove {:.1f}% points from {} orders because too concentrated'.format(100 * (1 - keep.mean()),
                                                                                 len(orders_too_concentrated)))
    df = df[keep]

    return df


def resample_traj(df, resample_time=20):
    if df.empty:
        return df

    init_len = len(df.index)
    index_lst = [0]
    df.reset_index(inplace=True, drop=True)
    for i in df.index[:-1]:
        if i in index_lst:
            check = df.time[i + 1:].gt(df.time[i] + timedelta(seconds=resample_time))
            if all(~check):
                break
            index_lst.append(check.idxmax())

    df = df.iloc[index_lst].reset_index(drop=True)

    print('remove {:.1f}% of points after {} sec resampling'.format(
        100 * (init_len - len(df.index)) / init_len, resample_time, resample_time))
    return df


def filter_gps(df_all, n_sample_trajs, dataset_name, resample_time=40, min_distance=80,
               traj_min_len=20, keep_every=1, save=False, plt_path=False, filter_data=False):
    # df_all = df_all.head(30)
    _traj_gb = list(df_all.groupby('traj_id'))[:n_sample_trajs]
    if not filter_data:
        return [traj_df for _, traj_df in _traj_gb]

    print('Number of trajectories {}'.format(len(_traj_gb)))
    traj_gb = []
    excluded_traj = []
    for _, traj_df in tqdm(_traj_gb, disable=False):
        traj_df.reset_index(drop=True)
        if dataset_name == 'Geolife':
            traj_df.sort_values(by='time', inplace=True)

        traj_id = set(traj_df['traj_id']).pop()
        print('Process traj id {} init len {}'.format(traj_id, len(traj_df)))

        # if not traj_id == 20080501001976:
        #     continue

        Path(f"viz_debug/{dataset_name}/{timestamp}/original").mkdir(parents=True, exist_ok=True)
        # print('Process traj id {} with mean freq {}'.format(traj_id, freq))
        show_trace(traj_df, traj_id,
                   filename=f"viz_debug/{dataset_name}/{timestamp}/original/{traj_id}.html",
                   save=save,
                   dataset_name=dataset_name,
                   plt_path=plt_path)

        # Step 1: Process time
        # traj_df = rm_time_close(traj_df, min_time_delta=freq)
        # show_trace(traj_df, traj_id, filename="rm_time_close.html", plt_path=plt_path)

        # Step 2: Keep a GPS point every some seconds (resample_time)
        if dataset_name == 'Geolife':
            Path(f"viz_debug/{dataset_name}/{timestamp}/filtered/downsampled").mkdir(parents=True, exist_ok=True)
            traj_df = resample_traj(traj_df, resample_time=resample_time)
            show_trace(traj_df, traj_id,
                       filename=f"viz_debug/{dataset_name}/{timestamp}/filtered/downsampled/{traj_id}.html",
                       save=save,
                       dataset_name=dataset_name,
                       plt_path=plt_path)

        # Step 3.1: Process distance
        Path(f"viz_debug/{dataset_name}/{timestamp}/filtered/rm_in_diameter").mkdir(parents=True, exist_ok=True)
        traj_df = rm_in_diameter(traj_df, path_min_diameter=50)
        show_trace(traj_df, traj_id,
                   filename=f"viz_debug/{dataset_name}/{timestamp}/filtered/rm_in_diameter/{traj_id}.html",
                   save=save,
                   dataset_name=dataset_name,
                   plt_path=plt_path)

        # Step 3.2: Process distance
        Path(f"viz_debug/{dataset_name}/{timestamp}/filtered/rm_dist_close").mkdir(parents=True, exist_ok=True)
        traj_df = rm_dist_close(traj_df, min_distance=min_distance, max_distance=500)
        show_trace(traj_df, traj_id,
                   filename=f"viz_debug/{dataset_name}/{timestamp}/filtered/rm_dist_close/{traj_id}.html",
                   save=save,
                   dataset_name=dataset_name,
                   plt_path=plt_path)

        # Step 4: Keep a GPS point every some (keep_every) GPS
        Path(f"viz_debug/{dataset_name}/{timestamp}/filtered/keep_every").mkdir(parents=True, exist_ok=True)
        traj_df = traj_df.iloc[::keep_every]
        show_trace(traj_df, traj_id,
                   filename=f"viz_debug/{dataset_name}/{timestamp}/filtered/keep_every/{traj_id}.html",
                   save=save,
                   dataset_name=dataset_name,
                   plt_path=plt_path)

        # Step 5: Keep trajectories with at least traj_min_len length
        print('Process traj id {} final len {}'.format(traj_id, len(traj_df)))
        if traj_df.empty or len(traj_df) < traj_min_len:
            excluded_traj.append(traj_id)
            print('traj id {} len {} excluded'.format(traj_id, len(traj_df)))
            continue

        # Save final traj
        Path(f"viz_debug/{dataset_name}/{timestamp}/filtered/final").mkdir(parents=True, exist_ok=True)
        show_trace(traj_df, traj_id,
                   filename=f"viz_debug/{dataset_name}/{timestamp}/filtered/final/{traj_id}.html",
                   save=save,
                   dataset_name=dataset_name,
                   plt_path=plt_path)

        traj_gb.append(traj_df)

    print("Number of excluded traj: {}".format(len(excluded_traj)))
    with open(f'viz_debug/{dataset_name}/{timestamp}/excluded traj.txt', 'w') as f:
        json.dump(excluded_traj, f, indent=4)

    return traj_gb


def save_traj(traj_gb, dataset_name):
    filepath = os.path.join('Datasets', dataset_name, 'meta')
    torch.save(traj_gb, os.path.join(filepath, 'traj_gb.pt'))


def plt_degree(G):
    print('Plot degree...')
    import collections

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b', align='center')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(deg)

    # # draw graph in inset
    # plt.axes([0.4, 0.4, 0.5, 0.5])
    # # Gcc = sorted(nx.subgraphs(G), key=len, reverse=True)[0]
    # pos = nx.spring_layout(G)
    # plt.axis('off')
    # nx.draw_networkx_nodes(G, pos, node_size=20)
    # nx.draw_networkx_edges(G, pos, alpha=0.4)

    plt.show()


def main():
    # # Warning: Load and save raw data. Only once!!
    datasets = ['Geolife', 'iWet']
    dataset_name = datasets[1]

    init_gps_data = False
    if init_gps_data:
        if dataset_name == 'Geolife':
            # https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
            df = read_all_users_geolife('Datasets/Geolife/GeolifeTrajectories1.3/Data')
            df.to_pickle('Datasets/Geolife/GeolifeTrajectories1.3/geolife.pkl')
        elif dataset_name == 'iWet':
            df = read_all_iwet("Datasets/iWet/iWetTrajectories/iWetDetails")
            df.to_pickle('Datasets/iWet/iWetTrajectories/iWetDetails.pkl')

    coords = {
        # "Geolife": {'latMin': 39.664, 'latMax': 40.131, 'longMin': 115.923, 'longMax': 116.86} # big region
        # "Geolife": {'latMin': 39.877, 'latMax': 39.952, 'longMin': 116.302, 'longMax': 116.451}, # sample
        "Geolife": {'latMin': 39.886, 'latMax': 39.984, 'longMin': 116.293, 'longMax': 116.488},  # smaller region
        "iWet": {'latMin': 40.566, 'latMax': 40.711, 'longMin': 22.793, 'longMax': 23.086}
    }

    latMin = coords[dataset_name]['latMin']
    latMax = coords[dataset_name]['latMax']
    longMin = coords[dataset_name]['longMin']
    longMax = coords[dataset_name]['longMax']

    if dataset_name == 'Geolife':
        df_all = read_geolife_gps('Datasets/Geolife/GeolifeTrajectories1.3/geolife.pkl',
                                  latMin, latMax, longMin, longMax, min_len=100)

        max_time_gap = 2 * 60  # seconds
        same_order = df_all.traj_id.diff() != 1
        big_time_gap = pd.to_datetime(df_all.time).diff() > np.timedelta64(max_time_gap, 's')
        should_split = big_time_gap & same_order
        df_all.traj_id += should_split.cumsum()
        print('split paths at {} position because longer time delta than {}s'.format(should_split.sum(), max_time_gap))
    else:
        # Unfortunately, because of copyright reasons, this dataset is not available to the public.
        df_all = read_iwet_gps('Datasets/iWet/iWetTrajectories/iWetDetails.pkl', latMin, latMax, longMin, longMax)

    df_all['traj_id'] = df_all['traj_id'].astype(int)

    df_all = df_all.groupby('traj_id').filter(lambda x: len(x) > 50)

    # latMin = df_all['lat'].min()
    # latMax = df_all['lat'].max()
    # longMin = df_all['lon'].min()
    # longMax = df_all['lon'].max()

    print('latMin: {} latMax: {} longMin: {} longMax: {}'.format(latMin, latMax, longMin, longMax))

    """
        Graph related computations
    """

    # # Download Graph
    # Set download to True to download the map
    org_G = map_graph(latMin, latMax, longMin, longMax, dataset_name, download=False, plt_graph=False)

    # mapped_nodes_id = dict((osm_id, i) for i, (osm_id, _) in enumerate(G.nodes(data=True)))
    G = nx.convert_node_labels_to_integers(org_G, ordering="default")  # ordering="default"
    # plt_degree(G)

    prep_graph_stats = False
    if prep_graph_stats:
        # Save shortest path distance matrix
        # Set save to True to save shortest-path-distance-matrix.pt
        shortest_path_distance_matrix(G, dataset_name=dataset_name, save=True)

        # Save graph shape files for fmm
        # Set save to True to save "nodes.shp" and "edges.shp"
        graph_shapefile_directional(G, dataset_name=dataset_name, save=True)

        # Save graph nodes and edges
        # Set save to True to save  "nodes.txt" and "edges.txt"
        graph_node_edges(G, dataset_name=dataset_name, save=True)

    """
        GPS related computations
    """
    prep_traj = True
    if prep_traj:
        # # Filter gps data
        resamle_time = 10
        keep_every = 10
        n_sample_trajs = None  # !!!!!!!!!! Set to None to process all trajectories. !!!!!!!!!!
        # This functions applies a difference range of filters in the GPS data. Please see comments inside.
        traj_gb = filter_gps(df_all,
                             n_sample_trajs=n_sample_trajs,
                             resample_time=resamle_time,
                             min_distance=50,
                             traj_min_len=6,  # 10,
                             dataset_name=dataset_name,
                             keep_every=keep_every,
                             save=True,
                             plt_path=False,
                             filter_data=True)

        # Set save to True to save paths.txt. This file is needed for Gretel algorithm [1].
        # Keep both FastMap and create_ubodt False or True based on the fast map matching algorithm [2.]
        # G and org_G are used for debug purposes
        traj_gb = mapped_paths(G=G, org_G=org_G, traj_gb=traj_gb,
                               FastMap=False, dataset_name=dataset_name,
                               create_ubodt=False, save=True, plt_paths=False)

        # # Set save to True to save lengths.txt. This file is needed for Gretel algorithm [1].
        save_lengths(traj_gb, dataset_name)

        # Set save to True to save observations.txt. This file is needed for Gretel algorithm [1].
        save_weights(traj_gb, dataset_name, rbf_sigma=5e-4, save=True, plt_weights=False)


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    main()
