"""Module for individual-based compartmental infection model for premises with HPAI."""
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.path import Path
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.lines import Line2D
import copy
import geopandas as gpd
from shapely.geometry import Point
# from numba import njit


class DataFilePaths:
    """
    Contains file paths required for data processing operations.

    This class is stores file paths related to data categories.

    Attributes:
        premises: str
            File path associated with premises data.
        cases: str
            File path associated with cases data.
        match: str
            File path associated with match data.
        regions: Optional[str]
            File path associated with regions data, if provided.
        counties: Optional[str]
            File path associated with counties data, if provided.
    """
    def __init__(self, premises, cases, match, regions=None, counties=None):
        self.premises = premises
        self.cases = cases
        self.match = match
        self.regions = regions
        self.counties = counties

class DataLoader:
    """
    Manages and processes data related to poultry premises, cases, and geographic
    regions or counties.

    This class handles data loading from various file types.

    Attributes:
        data_file_paths (object): Stores paths to data files (e.g., premises, cases, match,
            regions, counties) required for processing.
        date_start (datetime): The starting date defining the analysis period.
        date_end (datetime): The ending date defining the analysis period.
        select_region (Optional[int]): Optionally specifies a regional filter to isolate
            premises data for a specific region by its index.
        select_county (Optional[int]): Optionally specifies a county filter to isolate
            premises data for a specific county by its index.
        location (numpy.ndarray): (X,Y)-coordinates of premises locations.
        poultry_numbers (numpy.ndarray): Matrix representing poultry counts for multiple
            species per premises.
        infected_premises (numpy.ndarray): Indices of premises reported as infected.
        end_day (int): Number of days between the analysis start and end dates.
        report_day (numpy.ndarray): Days since the start date corresponding to report
            dates.
        region (Optional[numpy.ndarray]): Region indices associated with premises locations,
            if regions data is provided.
        region_names (Optional[list of str]): Names of all regions.
        region_names_ab (Optional[list of str]): Abbreviations of all region names.
        polygons_r (Optional[geopandas.GeoDataFrame]): Polygonal data representing regions
            for spatial relationships.
        county (Optional[numpy.ndarray]): County indices associated with premises locations,
            if counties data is provided.
        county_names (Optional[list of str]): Names of all counties.
        polygons_c (Optional[geopandas.GeoDataFrame]): Polygonal data representing counties
            for spatial relationships.
        included_premises (Optional[numpy.ndarray]): List of premises indices included after
            applying region/county filter.
    """
    def __init__(self, data_file_paths, date_start, date_end, select_region=None, select_county=None):
        self.data_file_paths = data_file_paths
        self.date_start = date_start
        self.date_end = date_end
        self.select_region = select_region
        self.select_county = select_county
        if self.select_region is not None and self.select_county is not None:
            raise ValueError("Only one of select_region or select_county can be specified.")
        if self.select_region is not None and self.data_file_paths.regions is None:
            raise ValueError("Regions file path must be provided when selecting a region.")
        if self.select_county is not None and self.data_file_paths.counties is None:
            raise ValueError("Counties file path must be provided when selecting a county.")

        # Load premises data
        try:
            data_premises = np.loadtxt(self.data_file_paths.premises)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing premises file: {self.data_file_paths.premises}")
        self.location = data_premises[:, 2:4].T
        self.poultry_numbers = data_premises[:, 4:].T.astype(int)

        # Load cases data
        data_cases = pd.read_excel(self.data_file_paths.cases)
        self.infected_premises = np.loadtxt(self.data_file_paths.match)
        # Convert dates to days since reference date.
        report_date = pd.to_datetime(data_cases['ReportDate'])
        self.end_day = (self.date_end - self.date_start).days
        self.report_day = np.array((report_date - self.date_start).dt.days.values)
        # Remove premises that are reported after the end date
        self.infected_premises = self.infected_premises[(self.report_day <= self.end_day)].astype(int)
        self.report_day = self.report_day[(self.report_day <= self.end_day)]
        # Remove secondary infections of premises in the time period
        unique_premises, counts = np.unique(self.infected_premises, return_counts=True)
        non_unique_premises = unique_premises[np.where(counts > 1)[0]]
        if len(non_unique_premises) > 0:
            remove_indices = np.concatenate(
                [np.where(self.infected_premises == value)[0][:-1] for value in non_unique_premises if
                 value in self.infected_premises])
            keep_indices = np.setdiff1d(np.arange(len(self.infected_premises)), remove_indices)
            self.infected_premises = self.infected_premises[keep_indices]
            self.report_day = self.report_day[keep_indices]

        # Load regions data if provided
        if self.data_file_paths.regions is not None:
            try:
                polygons = gpd.read_file(self.data_file_paths.regions)
                polygons = polygons[:-1]
                points_list = [Point(1000 * self.location[0, i], 1000 * self.location[1, i]) for i in range(self.location.shape[1])]
                points_gdf = gpd.GeoDataFrame(geometry=points_list, crs=polygons.crs)
                points_with_polygons = gpd.sjoin(points_gdf, polygons, how='left', predicate='within')
                self.region = points_with_polygons.index_right.values
                nan_regions = np.where(np.isnan(self.region))[0]
                for i in range(np.sum(np.isnan(self.region))):
                    self.region[nan_regions[i]] = np.argmin(polygons.distance(points_list[nan_regions[i]]))
                self.region = self.region.astype(int)
                self.region_names = ['North East', 'North West', 'Yorkshire and the Humber', 'East Midlands',
                                     'West Midlands', 'East of England', 'London', 'South East', 'South West', 'Wales',
                                     'Scotland']
                self.region_names_ab = ['NE', 'NW', 'Y&H', 'EM', 'WM', 'EoE', 'LDN', 'SE', 'SW', 'WAL', 'SCT']
                self.polygons_r = polygons
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing premises file: {self.data_file_paths.regions}")
        if self.data_file_paths.counties is not None:
            try:
                polygons = gpd.read_file(self.data_file_paths.counties)
                polygons = polygons
                points_list = [Point(1000 * self.location[0, i], 1000 * self.location[1, i]) for i in range(self.location.shape[1])]
                points_gdf = gpd.GeoDataFrame(geometry=points_list, crs=polygons.crs)
                points_with_polygons = gpd.sjoin(points_gdf, polygons, how='left', predicate='within')
                self.county = points_with_polygons.index_right.values
                nan_counties = np.where(np.isnan(self.county))[0]
                for i in range(np.sum(np.isnan(self.county))):
                    self.county[nan_counties[i]] = np.argmin(polygons.distance(points_list[nan_counties[i]]))
                self.county = self.county.astype(int)
                self.county_names = polygons.CTYUA23NM.tolist()
                self.polygons_c = polygons
                self.county_names = np.delete(self.county_names, np.s_[153:164])
                self.polygons_c = self.polygons_c.drop(index=range(153, 164))
                self.polygons_c = self.polygons_c.reset_index(drop=True)
                self.county[self.county > 153] -= 11
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing premises file: {self.data_file_paths.counties}")

        # Remove data from other regions/counties if one select region/county is specified
        if select_region is not None:
            self.location = self.location[:, self.region == select_region]
            self.poultry_numbers = self.poultry_numbers[:, self.region == select_region]
            self.included_premises = np.arange(len(self.region))[self.region == select_region]
            if self.data_file_paths.counties is not None:
                self.county = self.county[self.region == select_region]
            self.region = self.region[self.region == select_region]
        elif select_county is not None:
            self.location = self.location[:, self.county == select_county]
            self.poultry_numbers = self.poultry_numbers[:, self.county == select_county]
            self.included_premises = np.arange(len(self.county))[self.county == select_county]
            if self.data_file_paths.regions is not None:
                self.region = self.region[self.county == select_county]
            self.county = self.county[self.county == select_county]
        if select_region is not None or select_county is not None:
            self.infected_premises = self.infected_premises[np.isin(self.infected_premises, self.included_premises)]
            self.report_day = self.report_day[np.isin(self.infected_premises, self.included_premises)]



class Data:
    """
    Represents a dataset with geographical and species-related data. Provides tools for processing,
    analyzing, and creating a grid for simulation purposes.

    The class is designed to handle raw datasets containing location, species population, and other
    relevant details to facilitate spatial or epidemiological analysis. It includes features for
    region and county data integration, as well as adaptive or fixed grid generation for simulations.
    Attributes like species numbers, population averages, and spatial indices are computed as part
    of the initialization.

    Attributes:
        location: ndarray
            (X,Y)-coordinates of premises.
        poultry_numbers: ndarray
            Population of poultry species per premises.
        date_start: Any
            Start date of the dataset.
        date_end: Any
            End date of the dataset.
        infected_premises: Any
            Dataset field for tracking infected premises. Type not explicitly specified.
        report_day: Any
            Dataset field for the day reports associated with cases. Type not explicitly specified.
        n_species: int
            Total number of species in the dataset.
        mean_premises_size: ndarray
            Mean population size for each species across premises.
        pop_over_mean: ndarray
            Calculated ratio of population numbers over their species means.
        n_premises: int
            Absolute number of premises.
        location_tree: cKDTree
            KDTree spatial representation of the premises for efficient queries.
        region: Any
            Region-level dataset information if available; otherwise None.
        region_names: Any
            Names of regions; populated if region-level data exists.
        region_names_ab: Any
            Abbreviated region names; applicable if region-level data is provided.
        polygons_r: Any
            Spatial polygon representations of regions, when available.
        county: Any
            County-level dataset information if available; otherwise None.
        county_names: Any
            Names of counties; populated when county-level data exists.
        polygons_c: Any
            Spatial polygon representations of counties, when available.
        grid_number: Optional[int]
            Number of grids specified or None.
        grid_size: Optional[int]
            Size of each grid square or None.
        adapt: bool
            Indicates whether adaptive grid generation is used.

    Methods:
        _set_up_grid:
            Configures the grid system for simulation. Depending on the adapt flag, the grid
            creation can vary between adaptive and non-adaptive methods.
    """

    def __init__(self, raw_data, grid_size=None, grid_number=None, adapt=False):

        # Load raw data
        self.location = raw_data.location
        self.poultry_numbers = raw_data.poultry_numbers
        self.date_start = raw_data.date_start
        self.date_end = raw_data.date_end
        self.infected_premises = raw_data.infected_premises
        self.report_day = raw_data.report_day
        self.select_region = raw_data.select_region
        self.select_county = raw_data.select_county
        self.end_day = raw_data.end_day

        # Process data
        self.n_species = self.poultry_numbers.shape[0]
        self.mean_premises_size = np.mean(self.poultry_numbers, axis=1)
        self.pop_over_mean = self.poultry_numbers / self.mean_premises_size[:, np.newaxis]
        self.n_premises = self.location.shape[1]
        self.location_tree = cKDTree(self.location.T)

        # Load regions and counties if available
        if raw_data.data_file_paths.regions is not None:
            self.region = raw_data.region
            self.region_names = raw_data.region_names
            self.region_names_ab = raw_data.region_names_ab
            self.polygons_r = raw_data.polygons_r
        else:
            self.region = None
            self.region_names = None
            self.region_names_ab = None
            self.polygons_r = None
        if raw_data.data_file_paths.counties is not None:
            self.county = raw_data.county
            self.county_names = raw_data.county_names
            self.polygons_c = raw_data.polygons_c
        else:
            self.county = None
            self.county_names = None
            self.polygons_c = None

        # Create grid for simulation either by grid size or grid number
        if adapt:
            if grid_size is not None:
                raise ValueError("grid_size cannot be specified when adapt is True.")
            if grid_number is None:
                grid_number = 10
        else:
            if (grid_size is None) and (grid_number is None):
                grid_size = 10
            elif (grid_size is not None) and (grid_number is not None):
                raise ValueError("Both grid_number and grid_size cannot be specified.")
        self.grid_number = grid_number
        self.grid_size = grid_size
        self.adapt = adapt

        # Set up grid for simulations
        self._set_up_grid()

    def _set_up_grid(self):

        # Get edges of UK coordinates
        min_x, max_x = np.min(self.location[0, :]), np.max(self.location[0, :])
        min_y, max_y = np.min(self.location[1, :]), np.max(self.location[1, :])
        uk_length = max(max_x - min_x, max_y - min_y)
        # Adaptive method
        if self.adapt:
            lambda_hat = self.n_premises / (self.grid_number ** 2)
            self.grid_size = [uk_length]
            # Grid location is bottom left corner
            if (max_x - min_x) > (max_y - min_y):
                self.grid_location_x = [min_x]
                self.grid_location_y = [min_y - ((max_x - min_x) - (max_y - min_y)) / 2]
            else:
                self.grid_location_x = [min_x - ((max_y - min_y) - (max_x - min_x)) / 2]
                self.grid_location_y = [min_y]
            self.premises_in_grid = [self.n_premises]
            # For each true entry in continue_adapt, split the corresponding grid into 4 smaller grids of equal size
            continue_adapt = [True]
            while any(continue_adapt):
                for idx in [i for i, c in enumerate(continue_adapt) if c]:
                    current_eq = (np.log(self.premises_in_grid[idx]) - np.log(lambda_hat)) ** 2
                    half_grid_size = self.grid_size[idx] / 2
                    tmp_size = np.full(4, half_grid_size)
                    offsets_x = (0, half_grid_size, 0, half_grid_size)
                    offsets_y = (0, 0, half_grid_size, half_grid_size)
                    tmp_x = np.array([self.grid_location_x[idx] + o for o in offsets_x])
                    tmp_y = np.array([self.grid_location_y[idx] + o for o in offsets_y])
                    x_lower, x_upper = tmp_x[:, None], (tmp_x + tmp_size)[:, None]
                    y_lower, y_upper = tmp_y[:, None], (tmp_y + tmp_size)[:, None]
                    in_grid = (
                            (x_lower <= self.location[0, :]) & (self.location[0, :] < x_upper) &
                            (y_lower <= self.location[1, :]) & (self.location[1, :] < y_upper)
                    )
                    tmp_premises = np.sum(in_grid, axis=1)
                    nonzero_grid = tmp_premises > 0
                    new_eq_tmp = np.log(tmp_premises[nonzero_grid]) - np.log(lambda_hat)
                    new_eq = np.dot(new_eq_tmp, new_eq_tmp) / np.count_nonzero(nonzero_grid)
                    if current_eq <= new_eq:
                        continue_adapt[idx] = False
                    else:
                        self.grid_size[idx:idx + 1] = tmp_size[nonzero_grid].tolist()
                        self.grid_location_x[idx:idx + 1] = tmp_x[nonzero_grid].tolist()
                        self.grid_location_y[idx:idx + 1] = tmp_y[nonzero_grid].tolist()
                        self.premises_in_grid[idx:idx + 1] = tmp_premises[nonzero_grid].tolist()
                        continue_adapt[idx:idx + 1] = [True] * np.count_nonzero(nonzero_grid)

            self.grid_size = np.array(self.grid_size)
            self.grid_location_x = np.array(self.grid_location_x)
            self.grid_location_y = np.array(self.grid_location_y)
            self.premises_in_grid = np.array(self.premises_in_grid)

            # Get number of grids and assign premises to grids
            self.n_grids = len(self.grid_size)
            self.premises_grid = np.empty(self.n_premises, dtype=int)
            grid_x1 = self.grid_location_x + self.grid_size
            grid_y1 = self.grid_location_y + self.grid_size
            for i in range(self.n_grids):
                in_x = (self.location[0, :] >= self.grid_location_x[i]) & (self.location[0, :] < grid_x1[i])
                in_y = (self.location[1, :] >= self.grid_location_y[i]) & (self.location[1, :] < grid_y1[i])
                self.premises_grid[in_x & in_y] = i

            # Precompute minimum distance between grids where distance is zero if connected
            centers_x = self.grid_location_x + self.grid_size / 2
            centers_y = self.grid_location_y + self.grid_size / 2
            half_sizes = self.grid_size / 2
            dx = np.subtract.outer(centers_x, centers_x)
            dy = np.subtract.outer(centers_y, centers_y)
            edge_x = np.maximum(0.0, np.abs(dx) - (half_sizes[:, None] + half_sizes[None, :]))
            edge_y = np.maximum(0.0, np.abs(dy) - (half_sizes[:, None] + half_sizes[None, :]))
            self.grid_dist2 = edge_x ** 2 + edge_y ** 2

        # Non-adaptive method
        else:
            if self.grid_number is not None:
                edges_x = np.linspace(min_x, min_x + uk_length, self.grid_number + 1)
                edges_y = np.linspace(min_y, min_y + uk_length, self.grid_number + 1)
            else:
                edges_x = np.arange(min_x, min_x + uk_length + self.grid_size, self.grid_size)
                edges_y = np.arange(min_y, min_y + uk_length + self.grid_size, self.grid_size)
            grid_x = np.digitize(self.location[0, :], edges_x) - 1
            grid_y = np.digitize(self.location[1, :], edges_y) - 1
            base_grid = grid_x + np.max(grid_x + 1) * grid_y
            unique_grid = np.unique(base_grid)
            grid_numbers = np.arange(0, len(unique_grid))
            if self.grid_number is not None:
                self.grid_size = (edges_x[1] - edges_x[0]) * np.ones(len(unique_grid))
            else:
                self.grid_size = self.grid_size * np.ones(len(unique_grid))

            # Assign premises to grid
            premises_in_grid = np.bincount(base_grid)
            birds_in_grid = np.bincount(base_grid, weights=np.sum(self.poultry_numbers, axis=0))
            birds_in_grid_species = np.zeros((self.n_species, len(birds_in_grid)))
            for i in range(self.n_species):
                birds_in_grid_species[i, :] = np.bincount(base_grid, weights=self.poultry_numbers[i, :])
            self.premises_in_grid = premises_in_grid[premises_in_grid > 0]
            self.birds_in_grid = birds_in_grid[birds_in_grid > 0]
            self.birds_in_grid_species = birds_in_grid_species[birds_in_grid_species > 0]

            # Grid location is bottom left corner
            self.grid_location_x = min_x + self.grid_size * (np.mod(unique_grid, np.max(grid_x + 1)))
            self.grid_location_y = min_y + self.grid_size * (np.floor(unique_grid / np.max(grid_x + 1)))
            self.premises_grid = np.array([grid_numbers[np.where(unique_grid == val)][0] for val in base_grid])
            self.n_grids = len(unique_grid)

            # Precompute minimum distance between grids where distance is zero if connected
            all_grid_x = self.grid_location_x[:, np.newaxis] - self.grid_location_x
            all_grid_y = self.grid_location_y[:, np.newaxis] - self.grid_location_y
            self.grid_dist2 = np.maximum(0, np.abs(all_grid_x) - self.grid_size) ** 2 + np.maximum(0, np.abs(
                all_grid_y) - self.grid_size) ** 2
        self.grid_dist2[self.grid_dist2 < np.min(self.grid_size)] = 0

    def plot_the_grid(self):
        """Plot the grid across the UK."""
        fig, ax = plt.subplots(figsize=(5, 8))
        uk = self.polygons_r.dissolve()
        uk.plot(ax=ax, facecolor='none', edgecolor="black", linewidth=0.5)
        ax.scatter(1000*self.location[0, :], 1000*self.location[1, :], s=3, edgecolors="w", linewidth=0.1)
        for i in range(self.n_grids):
            ax.add_patch(plt.Rectangle((1000*self.grid_location_x[i], 1000*self.grid_location_y[i]), 1000*self.grid_size[i],
                                       1000*self.grid_size[i], fill=False, edgecolor='r', lw=0.5))
        ax.set_xticks([])
        ax.set_yticks([])

class Parameter:
    def __init__(self, name, values, prior_type, prior_pars, fitted, description, sigma):
        self.name = name
        self.values = np.array(values, dtype=float)
        self.values_init = np.array(values, dtype=float)
        self.fitted = np.array(fitted, dtype=bool)
        self.prior_type = prior_type
        self.prior_pars = np.array(prior_pars, dtype=float)
        self.description = description
        self.sigma = np.array(sigma, dtype=float)
        self.mu = copy.deepcopy(self.log_values)

        # Validate prior types and shapes
        allowed_priors = {"gamma", "beta"}
        invalid = [p for p in self.prior_type if p not in allowed_priors]
        if invalid:
            raise ValueError(
                f"Parameter {name} has invalid prior types: {invalid}. "
                f"Allowed types are {sorted(allowed_priors)}."
            )
        if len(self.prior_type) != np.atleast_2d(self.values).shape[1]:
            raise ValueError(f"Parameter {name}: prior_type length mismatch.")
        if np.atleast_2d(self.prior_pars).shape != (np.atleast_2d(self.values).shape[1], 2):
            raise ValueError(f"Parameter {name}: prior_pars shape mismatch.")

    @property
    def log_values(self):
        """Return the log-transformed parameter values."""
        log_values = np.log(self.values)
        log_values[self.prior_type == 'beta'] = np.log(self.values[self.prior_type == 'beta'] / (1 - self.values[self.prior_type == 'beta']))
        return log_values

class ParameterSet:
    def __init__(self):
        """Container for model parameters."""
        self.epsilon = None
        self.gamma = None
        self.delta = None
        self.omega = None
        self.psi = None
        self.phi = None
        self.xi = None
        self.zeta = None
        self.nu = None
        self.rho = None

    def all_parameters(self):
        """Return all defined Parameter objects as a dict."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, Parameter)}

    def fitted_parameters(self):
        """Return only parameters that are fitted."""
        return {k: v for k, v in self.all_parameters().items() if np.any(v.fitted)}

class ModelStructure:
    """Class for holding the model structure and parameters."""
    def __init__(self, data, n_compartments=5, inf_compartments=None, data_compartments=None,
                 fixed_transitions=None, kernel_type='cauchy', par_values=None, par_values_init=None, par_fitted=None,
                 par_prior_type=None, par_prior_values=None, par_descriptions=None, par_sigma=None, transitions=None,
                 trans_priors_type=None, trans_priors_values=None):
        self.data = data
        self.n_species = data.n_species
        self.n_premises = data.n_premises
        self.n_compartments = n_compartments
        if inf_compartments is None:
            self.inf_compartments = np.array([False, False, True, True, False], dtype=bool)
        else:
            if len(inf_compartments) != self.n_compartments:
                raise TypeError("Invalid infectious list: must be of length n_compartments.")
            self.inf_compartments = inf_compartments
            for element in self.inf_compartments:
                if not isinstance(element, bool):
                    raise TypeError("Invalid infectious list: must an array of booleans.")
            if self.inf_compartments[0]:
                raise ValueError("Invalid infectious list: First compartment must not be infectious.")
        self.inf_compartments_idx = np.where(self.inf_compartments)[0]
        if data_compartments is None:
            self.data_compartments = np.array([False, False, False, True, False], dtype=bool)
        else:
            if len(data_compartments) != self.n_compartments:
                raise TypeError("Invalid infectious list: must be of length n_compartments.")
            self.data_compartments = data_compartments
            for element in self.data_compartments:
                if not isinstance(element, bool):
                    raise TypeError("Invalid infectious list: must an array of booleans.")
        self.data_compartments_idx = np.where(self.data_compartments)[0]
        if fixed_transitions is None:
            self.fixed_transitions = np.array([True, False, True], dtype=bool)
        else:
            if len(fixed_transitions) != self.n_compartments - 2:
                raise TypeError("Invalid fixed_transitions list: must be of length n_compartments - 2 (the number of "
                                "transitions between compartments excluding initial exposure).")
            if np.where(~fixed_transitions)[0] > self.data_compartments_idx - 2:
                raise TypeError("Invalid fixed_transitions list: non-fixed transition must occur before data compartments.")
            self.fixed_transitions = fixed_transitions
            for element in self.fixed_transitions:
                if not isinstance(element, bool):
                    raise TypeError("Invalid fixed_transitions list: must an array of booleans.")
            if np.sum(~self.fixed_transitions) != 1:
                raise ValueError("Invalid fixed_transitions list: there must be exactly one non-fixed transition.")
        self.kernel_type = kernel_type
        if self.kernel_type not in ['cauchy', 'exp']:
            raise ValueError("Invalid kernel type: must be 'cauchy' or 'exp'.")
        self.parameters = self.set_up_parameters(par_values, par_values_init, par_fitted, par_prior_type,
                                                 par_prior_values, par_descriptions, par_sigma)
        if transitions is None:
            self.transitions = [4, None, 3]
        else:
            if len(transitions) != self.n_compartments - 2:
                raise TypeError("Invalid transitions list: must be of length n_compartments - 2 (the number of "
                                "transitions between compartments excluding initial exposure).")
            for element in transitions:
                if not (isinstance(element, int) or element is None):
                    raise TypeError("Invalid transitions list: must be an array of integers or None.")
            if sum(1 for item in transitions if item is None) != 1:
                raise ValueError("Invalid transitions list: there must be exactly one non-fixed transition.")
            self.transitions = transitions
        if trans_priors_type is None:
            self.trans_priors_type = 'gamma'
        else:
            if trans_priors_type != 'gamma' and trans_priors_type != 'beta':
                raise ValueError(
                    f"Invalid trans_priors_type: {trans_priors_type}. Allowed types are 'gamma' or 'beta' only.")
            self.trans_priors_type = trans_priors_type
        if trans_priors_values is None:
            self.trans_priors_values = np.array([4.0, 2.0])
        else:
            if len(trans_priors_values) != 2:
                raise TypeError("Invalid trans_priors_values: must be the 2 prior inputs.")
            self.trans_priors_values = trans_priors_values
        self.non_fixed_transitions = np.zeros(self.data.infected_premises.shape[0])

    def kernel_function(self, distance2):
        """Calculate the kernel value for a given squared distance."""
        delta = self.parameters.delta.values
        if self.kernel_type == 'cauchy':
            omega = self.parameters.omega.values
            return cauchy_kernel(distance2, delta, omega)
        elif self.kernel_type == 'exp':
            return exp_kernel(distance2, delta)
        else:
            raise ValueError("kernel_type must be 'cauchy' or 'exp'.")

    def set_up_parameters(self, par_values, par_values_init, par_fitted, par_prior_type,
                            par_prior_values, par_descriptions, par_sigma):
        """Set up model parameters with provided or default values."""
        # List of default parameter values
        if par_values is None:
            par_values = [np.array([1e-5]), np.array([0.05] + [0.9] * (np.count_nonzero(self.inf_compartments) - 1)),
                              np.array([2.0]), np.array([1.33]), np.array([0.5] * self.n_species),
                              np.array([0.5] * self.n_species), np.array([1] + [0.5] * (self.n_species - 1)),
                              np.array([1] + [0.5] * (self.n_species - 1)), np.array([1.5, 0.5])]
        else:
            self.parameter_check(par_values, 'par_values')
        # Initial values are the same as par_values if not provided
        if par_values_init is None:
            par_values_init = copy.deepcopy(par_values)
        else:
            self.parameter_check(par_values_init, 'par_values_init')
        # Set which parameters are fitted
        if par_fitted is None:
            par_fitted = [np.array([True]), np.array([True] * np.count_nonzero(self.inf_compartments)),
                          np.array([True]), np.array([False]), np.array([True] * self.n_species),
                          np.array([True] * self.n_species), np.array([False] + [True] * (self.n_species - 1)),
                          np.array([False] + [True] * (self.n_species - 1)), np.array([True, True])]
        else:
            self.parameter_check(par_fitted, 'par_fitted')
        # Set prior types (gamma or beta)
        if par_prior_type is None:
            par_prior_type = [['gamma'], ['gamma'] * np.count_nonzero(self.inf_compartments),
                               ['gamma'], ['gamma'], ['beta'] * self.n_species, ['beta'] * self.n_species,
                               ['gamma'] * self.n_species, ['gamma'] * self.n_species, ['gamma', 'beta']]
        else:
            self.parameter_check(par_fitted, 'par_fitted')
        # Set prior parameter values
        if par_prior_values is None:
            par_prior_values = [np.array([[1, 1e-5]]),
                                 np.array([[1, 0.01]] + [[1, 0.8]] * (np.count_nonzero(self.inf_compartments) - 1)),
                                 np.array([[2.0, 1.0]]),
                                 np.array([[200, 1/150]]),
                                 np.array([[2.0, 2.0]] * self.n_species),
                                 np.array([[2.0, 2.0]] * self.n_species),
                                 np.array([[1.0, 1.0]] + [[2.0, 4.0]] * (self.n_species - 1)),
                                 np.array([[1.0, 1.0]] + [[2.0, 4.0]] * (self.n_species - 1)),
                                 np.array([[2.0, 2.0], [2.0, 2.0]])]
        else:
            self.parameter_check(par_prior_values, 'par_prior_values')
        # Set descriptions for parameters
        if par_descriptions is None:
            par_descriptions = [['Baseline infectious\npressure'],
                                ['Infectious pressure\nfrom infected premises',
                                 'Multiplicative factor\nfor notified premises'],
                                ['Scale parameter in\ntransmission kernel'],
                                ['Exponent in\ntransmission kernel'],
                                ['Exponent for\ninfected Galliformes',
                                 'Exponent for\ninfected waterfowl',
                                 'Exponent for\ninfected other birds'],
                                ['Exponent for\nsusceptible Galliformes',
                                 'Exponent for\nsusceptible waterfowl',
                                 'Exponent for\nsusceptible other birds'],
                                ['None', 'Relative transmissibility\nof waterfowl to Galliformes',
                                 'Relative transmissibility\nof other birds to Galliformes'],
                                ['None', 'Relative susceptibility\nof waterfowl to Galliformes',
                                 'Relative susceptibility\nof other birds to Galliformes'],
                                ['Shape of seasonality', 'Timing of seasonality']]
        else:
            self.parameter_check(par_descriptions, 'par_descriptions')
        # Set sigma for proposal distributions
        if par_sigma is None:
            par_sigma = [np.array([0.2]), np.array([0.5] + [0.1] * (np.count_nonzero(self.inf_compartments) - 1)),
                              np.array([0.2]), np.array([0.05]), np.array([3.0] * self.n_species),
                              np.array([3.0] * self.n_species), np.array([1.0] * self.n_species),
                              np.array([1.0] * self.n_species), np.array([1.0, 1.0])]
        else:
            self.parameter_check(par_sigma, 'par_sigma')
        par_names = ['epsilon', 'gamma', 'delta', 'omega', 'psi', 'phi', 'xi', 'zeta', 'nu']
        parameters = ParameterSet()
        for i, name in enumerate(par_names):
            setattr(parameters, name,
                    Parameter(
                        name=name,
                        values=par_values_init[i],
                        fitted=par_fitted[i],
                        prior_type=par_prior_type[i],
                        prior_pars=par_prior_values[i],
                        description=par_descriptions[i],
                        sigma=par_sigma[i]
                    ))

        return parameters

    def parameter_check(self, input_list, name):
        """Check validity of parameter input lists."""
        par_lengths = np.array([1, np.count_nonzero(self.inf_compartments), 1, 1, self.n_species, self.n_species,
                            self.n_species, self.n_species, 2])
        if len(input_list) != 9:
            raise ValueError(f"{name} must be a list of length 9.")
        for i in range(9):
            if len(input_list[i]) != par_lengths[i]:
                raise ValueError(f"{name}[{i}] has incorrect length.")
        if name in ['par_values', 'par_values_init', 'par_sigma']:
            for i in range(9):
                for element in input_list[i]:
                    if not isinstance(element, (int, float)):
                        raise TypeError(f"{name}[{i}] must be an array of numbers.")
            if input_list[7][0] != 1 or input_list[8][0] != 1:
                raise ValueError(f"{name}: First component of xi or zeta must be 1.")
        elif name == 'par_fitted':
            for i in range(9):
                for element in input_list[i]:
                    if not isinstance(element, bool):
                        raise TypeError(f"{name}[{i}] must be an array of booleans.")
            if input_list[7][0] or input_list[7][1]:
                raise ValueError(f"{name}: First component of xi or zeta must not be fitted.")
        elif name == 'par_prior_type':
            allowed_priors = {"gamma", "beta"}
            for i in range(9):
                invalid = [p for p in input_list[i] if p not in allowed_priors]
                if invalid:
                    raise ValueError(
                        f"{name}[{i}] has invalid prior types: {invalid}. "
                        f"Allowed types are {sorted(allowed_priors)}."
                    )
        elif name == 'par_prior_values':
            par_shapes = [(1, 2), (np.count_nonzero(self.inf_compartments), 2), (1, 2), (1, 2),
                          (self.n_species, 2), (self.n_species, 2), (self.n_species, 2),
                          (self.n_species, 2), (2, 2)]
            for i in range(9):
                if np.array(input_list[i]).shape != par_shapes[i]:
                    raise ValueError(f"{name}[{i}] has incorrect shape.")

    def get_chain_string(self, iter, simulation=False):
        """Generate a string identifier for the model configuration for saving outputs."""
        self.chain_string = (self.data.date_start.strftime('%Y%m%d') + '_' +
                             self.data.date_end.strftime('%Y%m%d') + '_' + self.kernel_type)
        if not simulation:
            self.chain_string += '_' + str(iter)
        if self.parameters.rho is not None:
            self.chain_string += '_rho'
        if self.data.select_region is not None:
            self.chain_string += self.data.region_names[self.data.select_region].replace(" ", "_") + '_'
        if self.data.select_county is not None:
            self.chain_string += self.data.county_names[self.data.select_county].replace(" ", "_") + '_'
        # if simulation:
        #     if self.biosecurity:
        #         self.chain_string += ('bio' + str(self.biosecurity_level) + '_' + str(self.biosecurity_duration) + '_' +
        #                               str(self.biosecurity_zone) + '_')
        #     if self.vaccine is not None:
        #         self.chain_string += 'v' + str(self.vaccine['doses']) + '_' + self.vaccine['strategy'] + '_teams' + str(
        #             self.vaccine['teams']) + '_' + str(self.vaccine['max_team_distance']) + '_eff' + str(
        #             int(100 * self.vaccine['efficacy_s'][0])) + '_' + str(
        #             int(100 * self.vaccine['efficacy_t'][0])) + '_'
        #         if self.vaccine['silent'] > 0:
        #             self.chain_string += 'silent' + str(self.vaccine['silent']) + '_'

    def get_season_times(self, t):
        """Calculate seasonal scaling factor at time t."""
        return np.exp(-self.parameters.nu.values[0] * (1 + np.cos(2 * np.pi * ((t + self.data.date_start.timetuple().tm_yday) / 365 - self.parameters.nu.values[1]))))

class ModelFitting:
    """Placeholder for the infection model fitting class."""
    def __init__(self, model_structure, chain_number=0, total_iterations=211000, single_iterations=1000,
                 burn_in=11000, occult_updates=0.05):
        # The burn-in period must be less than total iterations
        if burn_in >= total_iterations:
            raise ValueError("burn_in must be less than total_iterations.")
        # The number of single parameter updates must be less than or equal to total iterations
        if single_iterations > total_iterations:
            raise ValueError("single_iterations must be less than or equal to total_iterations.")
        self.model_structure = model_structure
        self.chain_number = chain_number
        self.total_iterations = total_iterations
        self.single_iterations = single_iterations
        self.burn_in = burn_in
        self.occult_updates = occult_updates
        self.non_fixed_transitions = model_structure.non_fixed_transitions[model_structure.data.report_day > -self.model_structure.transitions[self.model_structure.data_compartments_idx[0] - 1]]
        self.infected_premises = model_structure.data.infected_premises[model_structure.data.report_day > -self.model_structure.transitions[self.model_structure.data_compartments_idx[0] - 1]]
        self.report_day = model_structure.data.report_day[model_structure.data.report_day > -self.model_structure.transitions[self.model_structure.data_compartments_idx[0] - 1]]
        self.infected_premises_past = model_structure.data.infected_premises[model_structure.data.report_day <= -self.model_structure.transitions[self.model_structure.data_compartments_idx[0] - 1]]
        self.report_day_past = model_structure.data.report_day[model_structure.data.report_day <= -self.model_structure.transitions[self.model_structure.data_compartments_idx[0] - 1]]
        self.n_cases = len(self.infected_premises)
        self.end_day = model_structure.data.end_day

        # MCMC tuning parameters
        self.cov_rate = 10
        self.lambda_rate = 50
        self.lambda_iter = 1

        # Storage for MCMC chains
        self.neg_log_posterior_chain = np.zeros(self.total_iterations + 1)
        self.parameter_chain = np.zeros((sum(np.sum(v.fitted) for v in self.model_structure.parameters.fitted_parameters().values()), self.total_iterations + 1))
        self.premises_chain = [None for _ in range(self.total_iterations + 1)]
        self.transition_chain = [None for _ in range(self.total_iterations + 1)]
        self.neg_log_posterior_chains = None
        self.parameter_chains = None
        self.premises_chains = None
        self.transition_chains = None
        self.neg_log_posterior_posterior = None
        self.parameter_posterior = None
        self.premises_posterior = None
        self.transition_posterior = None

        self._cached_parameters = None
        self._cached_infected_premises = None
        self._cached_non_fixed_transitions = None
        self._cached_first_exposure = None
        self._cached_neg_log_likelihood = None

        # Initial chain string
        self.model_structure.get_chain_string(self.total_iterations)

    @property
    def exposure_day(self):
        """Calculate exposure day for infected premises based on report day and transitions."""
        return self.report_day - (sum(x for x in self.model_structure.transitions[:(self.model_structure.data_compartments_idx[0] - 1)] if x is not None) + np.round(self.non_fixed_transitions).astype(int))

    def run_mcmc_chain(self, save_iter=None):
        """Run the MCMC fitting procedure."""
        # Set up saving MCMC after given number of iterations
        if save_iter is None:
            save_iter = np.array([self.total_iterations])
        self.n_to_fit = sum(np.sum(v.fitted) for v in self.model_structure.parameters.fitted_parameters().values())
        init_pars = copy.deepcopy(self.model_structure.parameters)
        # Get initial random non-fixed transitions
        if self.model_structure.trans_priors_type == 'gamma':
            self.non_fixed_transitions = np.random.gamma(self.model_structure.trans_priors_values[0],
                                                       self.model_structure.trans_priors_values[1],
                                                       len(self.infected_premises))
        elif self.model_structure.trans_priors_type == 'beta':
            self.non_fixed_transitions = np.random.beta(self.model_structure.trans_priors_values[0],
                                                      self.model_structure.trans_priors_values[1],
                                                      len(self.infected_premises))

        # Initialise acceptance rates
        accept = 0
        accept_notification = np.zeros(3)
        update_notification = np.zeros(3)
        current_neg_log_post = self.get_neg_log_likelihood()
        # Save initial state to the first entry in the chain
        self.update_chain(current_neg_log_post)

        # Begin MCMC iterations
        for iter in range(self.total_iterations):
            print(f"MCMC iteration {iter + 1} of {self.total_iterations}")
            # Check if we need to save this iteration and get string
            if np.isin(iter + 1, save_iter):
                save = True
            else:
                save = False
            self.model_structure.get_chain_string(iter + 1)

            # Do initial single parameter updates
            if iter < self.single_iterations:
                for par_name, par in self.model_structure.parameters.fitted_parameters().items():
                    par_values = par.values[par.fitted]
                    par_log_values = par.log_values[par.fitted]
                    par_sigma = par.sigma[par.fitted]
                    for i in range(len(par_values)):

                        # Propose new value
                        old_value = par_values[i]
                        old_log_value = par_log_values[i]
                        new_log_value = old_log_value + np.random.normal(0, par_sigma[i])
                        new_value = self.get_exp(np.array([new_log_value]), np.array(par.prior_type)[par.fitted][i])[0]
                        par.values[np.where(par.fitted)[0][i]] = new_value
                        if np.array(par.prior_type)[par.fitted][i] == 'gamma':
                            neg_log_jacobian = -new_log_value + old_log_value
                        elif np.array(par.prior_type)[par.fitted][i] == 'beta':
                            neg_log_jacobian = -new_log_value + old_log_value - np.log(1 - new_value) + np.log(1 - old_value)

                        # Calculate new posterior
                        new_neg_log_likelihood = self.get_neg_log_likelihood()
                        new_neg_log_post = new_neg_log_likelihood + self.get_neg_log_prior()

                        # Accept or reject new value
                        if current_neg_log_post - new_neg_log_post - neg_log_jacobian > np.log(np.random.uniform()):
                            current_neg_log_post = new_neg_log_post
                            par.sigma[np.where(par.fitted)[0][i]] *= 1.4
                            accept += 1 / self.n_to_fit
                            # print(f"Iteration {iter + 1}: Accepted {par_name}_{i} from {old_value:.4f} to {new_value:.4f}, LL: {current_neg_log_post:.2f}")
                        else:
                            # print(f"Iteration {iter + 1}: Rejected {par_name}_{i} from {old_value:.4f} to {new_value:.4f}, LL: {current_neg_log_post:.2f}")
                            par.sigma[np.where(par.fitted)[0][i]] *= (1.4 ** -0.7857143)
                            par.values[np.where(par.fitted)[0][i]] = old_value
            else:
                # Continue with multivariate parameter update
                if iter == self.single_iterations:
                    for par_name, par in self.model_structure.parameters.fitted_parameters().items():
                        par.sigma = np.diag(getattr(init_pars,par_name).sigma)
                neg_log_jacobian = 0
                current_pars = copy.deepcopy(self.model_structure.parameters)

                # Propose new values by parameter block
                for par_name, par in self.model_structure.parameters.fitted_parameters().items():
                    sd = np.sqrt(np.diag(par.sigma[np.ix_(par.fitted, par.fitted)]))
                    denominator = np.outer(sd, sd)
                    denominator = np.where((denominator == 0) | np.isnan(denominator), 1e-16, denominator)
                    correlation = par.sigma[np.ix_(par.fitted, par.fitted)] / denominator
                    correlation = np.where((par.sigma[np.ix_(par.fitted, par.fitted)] == 0) | np.isnan(par.sigma[np.ix_(par.fitted, par.fitted)]), 0, correlation)
                    old_value = par.values[par.fitted]
                    old_log_value = par.log_values[par.fitted]
                    new_log_value = old_log_value + np.random.multivariate_normal(np.zeros(len(old_log_value)), correlation) * (1 + (iter % 2) * (self.lambda_iter - 1)) * 2.38 * sd / np.sqrt(self.n_to_fit)
                    new_value = self.get_exp(new_log_value, np.array(par.prior_type)[par.fitted])
                    par.values[np.where(par.fitted)[0]] = new_value
                    for i in range(len(new_value)):
                        if np.array(par.prior_type)[par.fitted][i] == 'gamma':
                            neg_log_jacobian += -new_log_value[i] + old_log_value[i]
                        elif np.array(par.prior_type)[par.fitted][i] == 'beta':
                            neg_log_jacobian += -new_log_value[i] + old_log_value[i] - np.log(1 - new_value[i]) + np.log(1 - old_value[i])

                # Calculate new posterior after all parameters changed
                new_neg_log_likelihood = self.get_neg_log_likelihood()
                new_neg_log_post = new_neg_log_likelihood + self.get_neg_log_prior()
                iter_2 = iter - self.single_iterations + 1

                # Accept or reject new parameter set
                if current_neg_log_post - new_neg_log_post - neg_log_jacobian > np.log(np.random.uniform()):
                    current_neg_log_post = new_neg_log_post
                    # print(f"Iteration {iter + 1}: Accepted, LL: {current_neg_log_post:.2f}")
                    accept += 1
                    if iter % 2 == 0:
                        self.lambda_iter *= (1 + self.lambda_rate / (self.lambda_rate + iter_2))
                else:
                    # print(f"Iteration {iter + 1}: Rejected, LL: {current_neg_log_post:.2f}")
                    for par_name, par in self.model_structure.parameters.fitted_parameters().items():
                        par.values[par.fitted] = getattr(current_pars, par_name).values[par.fitted]
                    if iter % 2 == 1:
                        self.lambda_iter *= ((1 + self.lambda_rate / (self.lambda_rate + iter_2)) ** -0.305483)

                # Update mean and covariance matrix
                for par_name, par in self.model_structure.parameters.fitted_parameters().items():
                    new_mu = (iter_2 / (iter_2 + 1)) * par.mu[par.fitted]  + par.log_values[par.fitted] / (iter_2 + 1)
                    ss_pars = np.outer(par.log_values[par.fitted], par.log_values[par.fitted])
                    ss_mu = np.outer(par.mu[par.fitted], par.mu[par.fitted])
                    ss_new_mu = np.outer(new_mu, new_mu)
                    par.mu[par.fitted] = copy.deepcopy(new_mu)
                    par.sigma[np.ix_(par.fitted, par.fitted)] = ((iter_2 - 1 + self.cov_rate) * par.sigma[np.ix_(par.fitted, par.fitted)] + iter_2 * ss_mu - (iter_2 + 1) * ss_new_mu + ss_pars) / (iter_2 + self.cov_rate)
                    par.sigma[np.ix_(par.fitted, par.fitted)]  = self.symmetric_pos_def(par.sigma[np.ix_(par.fitted, par.fitted)])

            # Update occult infection premises
            n_premises_updates = np.floor(self.occult_updates * len(self.infected_premises)).astype(int)
            # Three events types (change infection time, add occult infection, remove occult infection)
            event_type = np.random.randint(0, 3, n_premises_updates)
            event_types = np.bincount(event_type, minlength=3)
            update_notification += event_types

            # Propose new transition times if required
            new_trans_times = np.zeros(n_premises_updates)
            new_trans_times[event_type == 0] = np.random.gamma(self.model_structure.trans_priors_values[0], self.model_structure.trans_priors_values[1], event_types[0])
            if (self.model_structure.trans_priors_values[0] == 4) & (self.model_structure.trans_priors_values[1] == 2):
                new_trans_times[event_type == 1] = stats.truncnorm.rvs(-0.6567 / 8.17, np.inf, loc=0.6567, scale=8.17, size=np.sum(event_type == 1))
            else:
                raise ValueError("New occult infection times currently only supported for Gamma(4,2) distribution.")
            for i, event in enumerate(event_type):
                if event == 0:

                    # Change time to notification
                    update_idx = np.random.randint(0, self.n_cases)
                    old_transition = self.non_fixed_transitions[update_idx]

                    # Update transition time depending on if occult or not
                    if update_idx > self.n_cases:
                        new_trans_times[i] = stats.truncnorm.rvs(-0.6567 / 8.17, np.inf, loc=0.6567, scale=8.17)
                        proposal = (stats.gamma.cdf(old_transition, a=self.model_structure.trans_priors_values[0],
                                                    scale=self.model_structure.trans_priors_values[1]) /
                                    stats.gamma.cdf(new_trans_times[i], a=self.model_structure.trans_priors_values[0],
                                                    scale=self.model_structure.trans_priors_values[1]))
                    else:
                        proposal = (stats.gamma.pdf(old_transition, a=self.model_structure.trans_priors_values[0],
                                                    scale=self.model_structure.trans_priors_values[0]) /
                                    stats.gamma.pdf(new_trans_times[i], a=self.model_structure.trans_priors_values[0],
                                                    scale=self.model_structure.trans_priors_values[0]))
                    self.non_fixed_transitions[update_idx] = new_trans_times[i]

                    # Calculate new posterior
                    new_neg_log_like = self.get_neg_log_likelihood()
                    new_neg_log_post = new_neg_log_like + self.get_neg_log_prior()

                    # Accept or reject new transition time
                    if current_neg_log_post - new_neg_log_post - np.log(proposal) > np.log(np.random.uniform()):
                        current_neg_log_post = new_neg_log_post
                        # print(f"Iteration {iter + 1}.{i}: Accepted change, LL: {current_neg_log_post:.2f}")
                        accept_notification[event] += 1
                    else:
                        # print(f"Iteration {iter + 1}.{i}: Rejected change, LL: {current_neg_log_post:.2f}")
                        self.non_fixed_transitions[update_idx] = old_transition
                elif event == 1:
                    # Add occult infection
                    new_premises = np.random.choice(np.setdiff1d(range(self.model_structure.n_premises),
                                                             self.infected_premises))
                    self.infected_premises = np.append(self.infected_premises, new_premises)
                    self.non_fixed_transitions = np.append(self.non_fixed_transitions, new_trans_times[i])
                    self.report_day = np.append(self.report_day, self.end_day)

                    # Calculate new posterior
                    new_neg_log_like = self.get_neg_log_likelihood()
                    new_neg_log_post = new_neg_log_like + self.get_neg_log_prior()
                    proposal = (self.model_structure.n_premises - len(self.infected_premises) + 1) / (
                            (len(self.infected_premises) - self.n_cases) * stats.truncnorm.pdf(new_trans_times[i].reshape(-1, 1), -0.6567 / 8.17, np.inf, loc=0.6567, scale=8.17))

                    # Accept or reject new occult infection
                    if current_neg_log_post - new_neg_log_post + np.log(proposal) > np.log(np.random.uniform()):
                        current_neg_log_post = new_neg_log_post
                        # print(f"Iteration {iter + 1}.{i}: Accepted addition, LL: {current_neg_log_post:.2f}")
                        accept_notification[event] += 1
                    else:
                        # print(f"Iteration {iter + 1}.{i}: Rejected addition, LL: {current_neg_log_post:.2f}")
                        self.infected_premises = self.infected_premises[:-1]
                        self.non_fixed_transitions = self.non_fixed_transitions[:-1]
                        self.report_day = self.report_day[:-1]
                elif (event == 2) and (len(self.infected_premises) > self.n_cases):
                    # Remove occult infection
                    remove_idx = np.random.choice(range(self.n_cases, len(self.infected_premises)))
                    remove_premises = self.infected_premises[remove_idx]
                    remove_transitions = copy.deepcopy(self.non_fixed_transitions[remove_idx])
                    self.infected_premises = np.delete(self.infected_premises, remove_idx)
                    self.non_fixed_transitions = np.delete(self.non_fixed_transitions, remove_idx)
                    self.report_day = np.delete(self.report_day, remove_idx)

                    # Calculate new posterior
                    new_neg_log_like = self.get_neg_log_likelihood()
                    new_neg_log_post = new_neg_log_like + self.get_neg_log_prior()
                    proposal = ((len(self.infected_premises) - self.n_cases + 1) * stats.truncnorm.pdf(remove_transitions, -0.6567 / 8.17, np.inf, loc=0.6567, scale=8.17)) \
                               / (self.model_structure.n_premises - len(self.infected_premises))

                    # Accept or reject removal of occult infection
                    if current_neg_log_post - new_neg_log_post + np.log(proposal) > np.log(np.random.uniform()):
                        current_neg_log_post = new_neg_log_post
                        # print(f"Iteration {iter + 1}.{i}: Accepted removal, LL: {current_neg_log_post:.2f}")
                        accept_notification[event] += 1
                    else:
                        # print(f"Iteration {iter + 1}.{i}: Rejected removal, LL: {current_neg_log_post:.2f}")
                        self.infected_premises = np.insert(self.infected_premises, remove_idx, remove_premises)
                        self.non_fixed_transitions = np.insert(self.non_fixed_transitions, remove_idx, remove_transitions)
                        self.report_day = np.insert(self.report_day, remove_idx, self.end_day)

            # Update MCMC chain with current values and save if required
            self.update_chain(current_neg_log_post)
            if save:
                self.save_chain(iter + 1)

    def save_chain(self, length_of_chain, dir='../outputs/'):
        """Save the MCMC chains to files."""
        np.save(f'{dir}mcmc_chain_{self.model_structure.chain_string}_neg_log_post_{self.chain_number}.npy', self.neg_log_posterior_chain[:(length_of_chain + 1)])
        np.save(f'{dir}mcmc_chain_{self.model_structure.chain_string}_parameters_{self.chain_number}.npy', self.parameter_chain[:, :(length_of_chain + 1)])

        #Convert premises_chain and transition_chain to arrays before saving
        max_length = max(len(arr) for arr in self.premises_chain[:(length_of_chain + 1)])
        premises_array = np.full((max_length, length_of_chain + 1), np.inf, dtype=float)
        transition_array = np.full((max_length, length_of_chain + 1), np.inf, dtype=float)
        for i, arr in enumerate(self.premises_chain[:(length_of_chain + 1)]):
            premises_array[:len(arr), i] = arr
        for i, arr in enumerate(self.transition_chain[:(length_of_chain + 1)]):
            transition_array[:len(arr), i] = arr
        np.save(f'{dir}mcmc_chain_{self.model_structure.chain_string}_premises_{self.chain_number}.npy', premises_array)
        np.save(f'{dir}mcmc_chain_{self.model_structure.chain_string}_transitions_{self.chain_number}.npy', transition_array)

    def load_chains(self, chain_numbers, values_per_chain=1000, dir='../outputs/'):
        """Load MCMC chains from files."""
        n_chains = len(chain_numbers)
        n_parameters = sum(np.sum(v.fitted) for v in self.model_structure.parameters.fitted_parameters().values())
        self.neg_log_posterior_chains = np.zeros((n_chains, self.total_iterations + 1))
        self.parameter_chains = np.zeros((n_chains, n_parameters, self.total_iterations + 1))
        premises_chains_tmp = [None for _ in range(n_chains)]
        transition_chains_tmp = [None for _ in range(n_chains)]
        for i in range(n_chains):
            self.neg_log_posterior_chains[i] = np.load(f'{dir}mcmc_chain_{self.model_structure.chain_string}_neg_log_post_{chain_numbers[i]}.npy')
            self.parameter_chains[i] = np.load(f'{dir}mcmc_chain_{self.model_structure.chain_string}_parameters_{chain_numbers[i]}.npy')
            premises_chains_tmp[i] = np.load(f'{dir}mcmc_chain_{self.model_structure.chain_string}_premises_{chain_numbers[i]}.npy')
            transition_chains_tmp[i] = np.load(f'{dir}mcmc_chain_{self.model_structure.chain_string}_transitions_{chain_numbers[i]}.npy')
        max_length = max(arr.shape[0] for arr in premises_chains_tmp)
        self.premises_chains = np.inf * np.ones((n_chains, max_length, self.total_iterations + 1))
        self.transition_chains = np.inf * np.ones((n_chains, max_length, self.total_iterations + 1))
        for i in range(n_chains):
            max_length_i = premises_chains_tmp[i].shape[0]
            self.premises_chains[i, :max_length_i, :] = premises_chains_tmp[i]
            self.transition_chains[i, :max_length_i, :] = transition_chains_tmp[i]

        # Get posterior samples by values_per_chain after burn-in
        choose_samples = np.arange(self.burn_in + 1, self.total_iterations + 1,
                                   (self.total_iterations - self.burn_in - 1) / (values_per_chain - 1)).astype(int)
        self.neg_log_posterior_posterior = self.neg_log_posterior_chains[:, choose_samples].flatten()
        self.parameter_posterior = self.parameter_chains[:, :, choose_samples].transpose(0, 2, 1).reshape(n_chains * values_per_chain, n_parameters)
        self.premises_posterior = self.premises_chains[:, :, choose_samples].transpose(0, 2, 1).reshape(n_chains * values_per_chain, max_length)
        self.transition_posterior = self.transition_chains[:, :, choose_samples].transpose(0, 2, 1).reshape(n_chains * values_per_chain, max_length)

    def update_chain(self, current_neg_log_post):
        """Update the MCMC chain with the current parameter values and likelihoods."""
        iter = np.sum(self.neg_log_posterior_chain != 0)
        self.neg_log_posterior_chain[iter] = current_neg_log_post
        i = 0
        for par_name, par in self.model_structure.parameters.fitted_parameters().items():
            self.parameter_chain[i:(i+np.sum(par.fitted)), iter] = par.values[par.fitted]
            i += np.sum(par.fitted)
        self.premises_chain[iter] = copy.deepcopy(self.infected_premises)
        self.transition_chain[iter] = copy.deepcopy(self.non_fixed_transitions)

    def get_neg_log_likelihood(self, smart_update=False):
        """Calculate the negative log likelihood of the model."""
        current_parameters = np.concatenate([
            par.values.flatten()
            for par_name, par in self.model_structure.parameters.fitted_parameters().items()
        ])
        if smart_update and ((self._cached_parameters is not None) and (np.array_equal(current_parameters, self._cached_parameters))):
            if (np.array_equal(self.non_fixed_transitions, self._cached_non_fixed_transitions)) and (np.array_equal(self.infected_premises, self._cached_infected_premises)):
                return self._cached_neg_log_likelihood
            elif np.array_equal(self.infected_premises, self._cached_infected_premises):
                #Update changed transition
                first_exposure = np.min(self.exposure_day)
                updated_premises = self.infected_premises[np.where(self.non_fixed_transitions != self._cached_non_fixed_transitions)[0]]
                updated_transition = self.non_fixed_transitions[np.where(self.non_fixed_transitions != self._cached_non_fixed_transitions)[0]]
                old_transition = self._cached_non_fixed_transitions[np.where(self.non_fixed_transitions != self._cached_non_fixed_transitions)[0]]

                if old_transition < updated_transition:
                    if first_exposure < self._cached_first_exposure:
                        self._cached_first_exposure = first_exposure
                else:
                    if first_exposure > self._cached_first_exposure:
                        self._cached_first_exposure = first_exposure
            else:
                # Update removal/addition of infected premises
                first_exposure = np.min(self.exposure_day)
                add_premises = np.setdiff1d(self.infected_premises, self._cached_infected_premises)
                remove_premises = np.setdiff1d(self._cached_infected_premises, self.infected_premises)
                if len(add_premises) > 0:
                    if first_exposure < self._cached_first_exposure:
                        self._cached_first_exposure = first_exposure
                if len(remove_premises) > 0:
                    if first_exposure > self._cached_first_exposure:
                        self._cached_first_exposure = first_exposure

        else:
            # Full update
            for par_name, par in self.model_structure.parameters.fitted_parameters().items():
                if np.any(par.values < 0):
                    return np.inf
            transmissibility = np.sum(self.model_structure.parameters.xi.values[:, np.newaxis] * (self.model_structure.data.pop_over_mean[:, self.infected_premises] ** self.model_structure.parameters.psi.values[:, np.newaxis]), axis=0)
            susceptibility = np.sum(self.model_structure.parameters.zeta.values[:, np.newaxis] * (self.model_structure.data.pop_over_mean ** self.model_structure.parameters.phi.values[:, np.newaxis]), axis=0)
            first_exposure = np.min(self.exposure_day)
            season_times = self.model_structure.get_season_times(np.arange(first_exposure, self.model_structure.data.end_day + 1))
            all_season_times = np.sum(season_times) * (self.model_structure.n_premises - len(self.infected_premises))

            src = self.model_structure.data.location[:, self.infected_premises]  # (2, k)
            dst = self.model_structure.data.location  # (2, n)
            src_sq = np.sum(src ** 2, axis=0)[:, None]  # (k, 1)
            dst_sq = np.sum(dst ** 2, axis=0)[None, :]  # (1, n)
            cross = src.T @ dst  # (k, n)
            distances2 = np.maximum(0, src_sq + dst_sq - 2 * cross)
            kernel_values = self.model_structure.kernel_function(distances2)
            log_like_1 = 0
            log_like_2 = 0
            log_like_3 = 0

            t0 = self.model_structure.transitions[0]
            t2 = self.model_structure.transitions[2]
            exposure_days_i = self.exposure_day[:, np.newaxis]
            exposure_days_j = self.exposure_day[np.newaxis, :]
            report_days_j = self.report_day[np.newaxis, :]
            mask_I = (exposure_days_j + t0 <= exposure_days_i) & (exposure_days_i < report_days_j)
            mask_N = (report_days_j <= exposure_days_i) & (exposure_days_i < report_days_j + t2)
            mask_R = (report_days_j + t2 <= exposure_days_i)
            kernel_ij = kernel_values[:, self.infected_premises]  # (k, k) matrix: i <- j interaction
            np.fill_diagonal(kernel_ij, 0)
            trans_j = transmissibility[np.newaxis, :]  # (1, k)
            susc_i = susceptibility[self.infected_premises][:, np.newaxis]  # (k, 1)
            beta_ij = trans_j * susc_i * kernel_ij
            gamma_0 = self.model_structure.parameters.gamma.values[0]  # gamma_0
            gamma_1 = self.model_structure.parameters.gamma.values[0] * self.model_structure.parameters.gamma.values[1]  # gamma_1
            F_I = gamma_0 * beta_ij * mask_I  # (k, k)
            F_N = gamma_1 * beta_ij * mask_N  # (k, k)
            F_R = np.zeros_like(beta_ij)
            if self.model_structure.parameters.rho is not None:
                rho = self.model_structure.parameters.rho.values[0]
                decay_term = np.exp(-rho * (exposure_days_i - (report_days_j + t2)))
                F_R = gamma_1 * beta_ij * mask_R * decay_term  # (k, k)
            force_of_infection = np.sum(F_I + F_N + F_R, axis=1)  # (k,) vector
            season_times_i = self.model_structure.get_season_times(self.exposure_day)  # (k,) vector
            exposure_rate = self.model_structure.parameters.epsilon.values[0] * season_times_i + force_of_infection
            log_like_1 = np.sum(np.log(exposure_rate))
            for i in range(len(self.infected_premises)):
                beta_values = transmissibility[i] * susceptibility * kernel_values[i]
                # Get the number of days each premises is susceptible when the i-th premises is infectious
                days_I = np.full(self.model_structure.n_premises, np.round(self.non_fixed_transitions[i]).astype(int))
                days_I[self.infected_premises] = np.clip(self.exposure_day - (self.exposure_day[i] + t0), 0, np.round(self.non_fixed_transitions[i]).astype(int))
                days_N = np.full(self.model_structure.n_premises, t2)
                days_N[self.infected_premises] = np.clip(self.exposure_day - (self.report_day[i]), 0, t2)
                if (self.model_structure.parameters.rho is not None):
                    days_R = np.full(self.model_structure.n_premises, self.model_structure.data.end_day - (self.report_day[i] + t2))
                    days_R[self.infected_premises] = np.clip(self.exposure_day - (self.report_day[i] + t2), 0, self.model_structure.data.end_day - (self.report_day[i] + t0))
                    decay_R = beta_values[:, np.newaxis] * np.exp(-self.model_structure.parameters.rho.values * np.arange(self.report_day[i] + t2, self.end_day + 1)[np.newaxis, :])
                    cols = np.arange(decay_R.shape[1])  # shape (20,)
                    mask = cols[None, :] > days_R[:, None]
                    decay_R[mask] = 0
                    log_like_2 += -gamma_1 * np.sum(decay_R)
                log_like_2 += -gamma_0 * np.sum(beta_values * days_I)
                log_like_2 += -gamma_1 * np.sum(beta_values * days_N)
            notif_like = stats.gamma.pdf(self.non_fixed_transitions[:self.n_cases], a=self.model_structure.trans_priors_values[0], scale=self.model_structure.trans_priors_values[1])
            log_like_3 += np.sum(np.log(notif_like))
            if self.n_cases < len(self.infected_premises):
                log_like_3 += np.sum(np.log(1 - stats.gamma.cdf(self.non_fixed_transitions[self.n_cases:], a=self.model_structure.trans_priors_values[0], scale=self.model_structure.trans_priors_values[1])))
            log_like_2 += -self.model_structure.parameters.epsilon.values[0] * all_season_times
            neg_log_like = -(log_like_1 + log_like_2 + log_like_3)
            self._cached_parameters = current_parameters
            self._cached_neg_log_likelihoods = neg_log_like
            self._cached_infected_premises = copy.deepcopy(self.infected_premises)
            self._cached_non_fixed_transitions = copy.deepcopy(self.non_fixed_transitions)
            self._cached_first_exposure = first_exposure
            return neg_log_like

    def get_neg_log_prior(self):
        """Calculate the negative log prior of the model."""
        log_prior = 0
        for par_name, par in self.model_structure.parameters.fitted_parameters().items():
            par_values = par.values[par.fitted]
            par_prior = np.array(par.prior_type)[par.fitted]
            par_prior_values = par.prior_pars[par.fitted]
            for i in range(len(par_values)):
                if par_prior[i] == 'gamma':
                    log_prior += stats.gamma.logpdf(par_values[i], a=par_prior_values[i, 0], scale=par_prior_values[i, 1])
                elif par_prior[i] == 'beta':
                    log_prior += stats.beta.logpdf(par_values[i], a=par_prior_values[i, 0], b=par_prior_values[i, 1])
        return -log_prior

    def symmetric_pos_def(self, matrix):
        """Make a matrix symmetric positive definite"""
        if isinstance(matrix, (list, np.ndarray)) and len(matrix) > 1:
            matrix = (matrix + matrix.T) / 2
            e_vals, e_vecs = np.linalg.eig(matrix)
            if sum(e_vals < 0) != 0:
                s_vals = sum(e_vals[e_vals < 0]) * 2
                t_vals = (s_vals * s_vals * 100) + 1
                if e_vals[e_vals > 0].size == 0:
                    return matrix
                else:
                    p_vals = min(e_vals[e_vals > 0])
                n_vals = e_vals[e_vals < 0]
                nn_vals = p_vals * (s_vals - n_vals) * (s_vals - n_vals) / t_vals
                d_vals = copy.deepcopy(e_vals)
                d_vals[d_vals <= 0] = nn_vals
                d_vals = np.diag(d_vals)
                matrix = np.matmul(np.matmul(e_vecs, d_vals), e_vecs.T)
        return matrix

    def get_exp(self, values, type):
        """Get the exponentiated values based on prior type."""
        exp_values = np.exp(values)
        exp_values[type == 'beta'] = exp_values[type == 'beta'] / (1 + exp_values[type == 'beta'])
        return exp_values

class ModelSimulator:
    """Placeholder for the infection model simulation class."""
    def __init__(self, model_structure=None, model_fitting=None, reps=10000, initial_condition_type=0, sellke=False,
                 vaccine=None, biosecurity=None):
        if model_structure is None and model_fitting is None:
            raise ValueError('model_structure or model_fitting are required')
        if model_structure is not None:
            self.model_structure = model_structure
            self.model_structure.get_chain_string(0, simulation=True)
            self.parameter_posterior = np.zeros(sum(np.sum(v.fitted) for v in self.model_structure.parameters.fitted_parameters().values()))
            i = 0
            for par_name, par in self.model_structure.parameters.fitted_parameters().items():
                self.parameter_posterior[i:(i + np.sum(par.fitted))] = par.values[par.fitted]
                i += np.sum(par.fitted)
            self.parameter_posterior = self.parameter_posterior[np.newaxis, :]
            self.premises_posterior = self.model_structure.data.infected_premises
            self.premises_posterior = self.premises_posterior[np.newaxis, :]
            if self.model_structure.trans_priors_type == 'gamma':
                self.transition_posterior = np.random.gamma(self.model_structure.trans_priors_values[0],
                                                             self.model_structure.trans_priors_values[1],
                                                             self.premises_posterior.shape[1])
            elif self.model_structure.trans_priors_type == 'beta':
                self.transition_posterior = np.random.beta(self.model_structure.trans_priors_values[0],
                                                            self.model_structure.trans_priors_values[1],
                                                            self.premises_posterior.shape[1])

            self.transition_posterior = self.transition_posterior[np.newaxis, :]
            self.n_premises = self.model_structure.n_premises
            self.end_day = self.model_structure.data.end_day
        if model_fitting is not None:
            self.model_fitting = model_fitting
            self.model_structure = model_fitting.model_structure
            self.parameter_posterior = model_fitting.parameter_posterior
            self.premises_posterior = model_fitting.premises_posterior
            self.transition_posterior = model_fitting.transition_posterior
            self.n_premises = self.model_fitting.model_structure.n_premises
            self.end_day = self.model_fitting.model_structure.data.end_day
            self.n_cases = self.model_fitting.n_cases
        self.reps = reps
        self.initial_condition_type = initial_condition_type
        self.vaccine = vaccine
        self.biosecurity = biosecurity
        self.sellke = sellke
        self.exposure_day = [None for _ in range(reps)]
        self.report_day = [None for _ in range(reps)]
        self.infected_premises = [None for _ in range(reps)]
        if self.sellke:
            self.resistances = np.tile(np.random.exponential(1, (100, self.model_structure.n_premises)), np.ceil(self.reps/100))[:self.reps, :]
            if np.ceil(self.reps / 100) > self.parameter_posterior.shape[0]:
                self.post_idx = np.repeat(np.random.choice(range(self.parameter_posterior.shape[0]), size=np.ceil(self.reps / 100).astype(int), replace=True), 100)[:self.reps]
            else:
                self.post_idx = np.repeat(np.random.choice(range(self.parameter_posterior.shape[0]), size=np.ceil(self.reps / 100).astype(int), replace=False), 100)[:self.reps]
        else:
            self.post_idx = np.random.choice(range(self.parameter_posterior.shape[0]), size=self.reps, replace=True)

    def run_model(self, save_results=True):
        for rep in range(self.reps):
            print(f"Simulation {rep + 1} of {self.reps}")
            if self.sellke:
                self.cumulative_hazard = np.zeros(self.model_structure.n_premises)
            self.premises_status = np.zeros((self.end_day + 1, self.model_structure.n_premises), dtype=int)
            self.premises_status_days = np.zeros(self.n_premises, dtype=int)
            if (self.sellke and rep % 100 == 0) or not self.sellke:
                fit_a, _, fit_b = stats.gamma.fit(self.transition_posterior[self.post_idx[rep], :self.n_cases], floc=0)
                self.non_fixed_transitions = np.random.gamma(fit_a, fit_b, self.n_premises)
                self.non_fixed_transitions[self.premises_posterior[self.post_idx[rep], :self.n_cases].astype(int)] = (
                    self.transition_posterior)[self.post_idx[rep], :self.n_cases]
            self.get_initial_conditions(rep)
            self.transmissibility = np.sum(self.model_structure.parameters.xi.values[:, np.newaxis] * (
                        self.model_structure.data.pop_over_mean ** self.model_structure.parameters.psi.values[:, np.newaxis]), axis=0)
            self.susceptibility = np.sum(self.model_structure.parameters.zeta.values[:, np.newaxis] * (
                        self.model_structure.data.pop_over_mean ** self.model_structure.parameters.phi.values[:, np.newaxis]), axis=0)
            self.max_susceptibility_grid = np.zeros(self.model_structure.data.n_grids)
            for i in range(self.model_structure.data.n_grids):
                self.max_susceptibility_grid[i] = np.max(self.susceptibility[self.model_structure.data.premises_grid == i])
            self.max_transmission_grid = np.zeros(self.model_structure.data.n_grids)
            gamma_multiplier = copy.deepcopy(self.model_structure.parameters.gamma.values)
            gamma_multiplier[0] = 1
            self.gamma = self.model_structure.parameters.gamma.values[0] * gamma_multiplier
            for i in range(self.model_structure.data.n_grids):
                self.max_transmission_grid[i] = np.max(self.transmissibility[self.model_structure.data.premises_grid == i])
            self.max_transmission_grid *= np.max(self.gamma)
            kernel_grid = self.model_structure.kernel_function(self.model_structure.data.grid_dist2)
            self.u_ab = 1 - np.exp(-self.max_susceptibility_grid * self.max_transmission_grid[:, np.newaxis] * kernel_grid)
            self.max_rate_grid = self.max_susceptibility_grid * kernel_grid
            for day in range(self.end_day):
                self.update_day(day, rep)
            self.report_day[rep] = self.exposure_day[rep] + np.round(self.non_fixed_transitions[self.infected_premises[rep]]).astype(int) + self.model_structure.transitions[0]
        self.report_day_projections = np.concatenate(self.report_day)
        self.report_premises_projections = np.concatenate(self.infected_premises)
        self.report_rep_projections = np.array([])
        for i, reports in enumerate(self.report_day):
            report_length = len(reports)
            self.report_rep_projections = np.append(self.report_rep_projections, np.full(report_length, i, dtype=int))
        self.report_time_projections = self.report_day_projections - (np.concatenate(self.exposure_day) + self.model_structure.transitions[0])
        if save_results:
            self.save_projections()

    def update_day(self, day, rep):
        season_time = self.model_structure.get_season_times(day)
        if self.vaccine is not None:
            raise NotImplementedError("Vaccination not yet implemented in ModelSimulator.")
        elif self.biosecurity is not None:
            raise NotImplementedError("Biosecurity not yet implemented in ModelSimulator.")
        else:
            new_transmissibility = self.transmissibility
            new_susceptibility = self.susceptibility
        if self.sellke:
            self.cumulative_hazard += self.model_structure.parameters.epsilon.values[0] * season_time * (new_susceptibility / self.susceptibility)
            expose_event = self.cumulative_hazard > self.resistances[rep]
        else:
            expose_event = 1 - np.exp(-self.model_structure.parameters.epsilon.values[0] * season_time * (new_susceptibility / self.susceptibility)) > np.random.uniform(size=self.n_premises)
        other_events = np.zeros(self.n_premises, dtype=int)
        infected_premises_now = np.where((0 < self.premises_status[day, :]) & (self.premises_status[day, :] < 4))[0]
        infection_status = self.premises_status[day, infected_premises_now]
        move_compartment = np.zeros(len(infected_premises_now), dtype=bool)
        move_compartment[infection_status == 1] = self.premises_status_days[infected_premises_now[infection_status == 1]] >= self.model_structure.transitions[0] - 1
        move_compartment[infection_status == 2] = self.premises_status_days[infected_premises_now[infection_status == 2]] >= np.round(self.non_fixed_transitions[infected_premises_now[infection_status == 2]]).astype(int) - 1
        move_compartment[infection_status == 3] = self.premises_status_days[infected_premises_now[infection_status == 3]] >= self.model_structure.transitions[2] - 1
        multiple_move = ((self.premises_status[day, infected_premises_now] == 2) & (np.round(self.non_fixed_transitions[infected_premises_now]).astype(int) == 0))
        other_events[infected_premises_now[move_compartment]] = 1
        other_events[infected_premises_now[move_compartment & multiple_move]] = 2
        infectious_premises = np.where(np.isin(self.premises_status[day, :], self.model_structure.inf_compartments_idx))[0]
        if self.model_structure.parameters.rho is not None:
            max_decay_days = \
            np.where(np.exp(-(1 / self.model_structure.parameters.rho.values[0]) * np.arange(500)) > 1e-20)[0][-1]
            infectious_premises = np.append(infectious_premises, np.where(
                (self.premises_status[day, :] == 4) & (self.premises_status_days <= max_decay_days))[0])
        if self.sellke:
            for a in infectious_premises:
                susceptibility_mask = self.premises_status[day, :] == 0
                diffs = self.model_structure.data.location[:, susceptibility_mask] - self.model_structure.data.location[:, a][:, np.newaxis]
                distances2 = np.einsum('ij,ij->j', diffs, diffs)
                if self.biosecurity or self.vaccine is not None:
                    sim_susceptibility = new_susceptibility[susceptibility_mask]
                else:
                    sim_susceptibility = self.susceptibility[susceptibility_mask]
                lambda_ij = (sim_susceptibility * self.gamma[self.premises_status[day, a] - 2] *
                             new_transmissibility[a] * season_time * self.model_structure.kernel_function(distances2))
                self.cumulative_hazard[susceptibility_mask] += lambda_ij
            expose_event = self.cumulative_hazard > self.resistances[rep]
        else:
            infectious_grids, inf_in_grid = np.unique(self.model_structure.data.premises_grid[infectious_premises], return_counts=True)
            sus_in_grid = np.bincount(self.model_structure.data.premises_grid[self.premises_status[day, :] == 0], minlength=self.model_structure.data.n_grids)
            sus_grids = np.where(sus_in_grid > 0)[0]
            for a_i, a in enumerate(infectious_grids):
                N_a = infectious_premises[self.model_structure.data.premises_grid[infectious_premises] == a]
                n_a = len(N_a)
                w_ab = 1 - (1 - self.u_ab[a, :]) ** n_a
                for b in sus_grids[sus_grids != a]:
                    n_b = sus_in_grid[b]
                    n_sample = np.random.binomial(n_b, w_ab[b])
                    if n_sample > 0:
                        N_b = np.where((self.model_structure.data.premises_grid == b) & (self.premises_status[day, :] == 0))[0]
                        N_sample = np.random.choice(N_b, n_sample, replace=False)
                        K_ij = self.model_structure.kernel_function(np.sum((self.model_structure.data.location[:, N_a, np.newaxis] - self.model_structure.data.location[:, np.newaxis, N_sample]) ** 2, axis=0))
                        p_ij = np.zeros((n_a, n_sample))
                        if self.biosecurity or self.vaccine is not None:
                            sim_susceptibility = new_susceptibility[N_sample]
                        else:
                            sim_susceptibility = self.susceptibility[N_sample]
                        p_ij[self.premises_status[day, N_a] != 4, :] = (
                                1 - np.exp(-sim_susceptibility * self.gamma[self.premises_status[day, N_a[self.premises_status[day, N_a] != 4]] - 2, np.newaxis] *
                                new_transmissibility[N_a[self.premises_status[day, N_a] != 4], np.newaxis] *
                                season_time * K_ij[self.premises_status[day, N_a] != 4]))
                        if self.model_structure.parameters.rho is not None:
                            days_since = np.exp(-(1 / self.model_structure.parameters.rho.values[0]) * (1 + self.premises_status_days[
                                N_a[self.premises_status[day, N_a] == 4]]))[:, np.newaxis]
                            p_ij[self.premises_status[day, N_a] == 4, :] = 1 - np.exp(
                                -sim_susceptibility * self.gamma[self.premises_status[day, N_a[self.premises_status[day, N_a] == 4]] - 2, np.newaxis] *
                                new_transmissibility[N_a[self.premises_status[day, N_a] == 4], np.newaxis] * days_since * season_time * K_ij[
                                    self.premises_status[day, N_a] == 4])
                        p_aj = 1 - np.prod(1 - p_ij, axis=0)
                        exp_mask = N_sample[np.random.rand(n_sample) < p_aj / w_ab[b]]
                        expose_event[exp_mask] = True
                N_b = np.where((self.model_structure.data.premises_grid == a) & (self.premises_status[day, :] == 0))[0]
                n_b = sus_in_grid[a]
                K_ij = self.model_structure.kernel_function(np.sum((self.model_structure.data.location[:, N_a, np.newaxis] -
                                                                    self.model_structure.data.location[:, np.newaxis, N_b]) ** 2, axis=0))
                p_ij = np.zeros((n_a, n_b))
                if self.biosecurity or self.vaccine is not None:
                    sim_susceptibility = new_susceptibility[N_b]
                else:
                    sim_susceptibility = self.susceptibility[N_b]
                p_ij[self.premises_status[day, N_a] != 4, :] = 1 - np.exp(
                    -sim_susceptibility * self.gamma[self.premises_status[day, N_a[self.premises_status[day, N_a] != 4]] - 2, np.newaxis] *
                    new_transmissibility[N_a[self.premises_status[day, N_a] != 4], np.newaxis] * season_time * K_ij[self.premises_status[day, N_a] != 4])
                if self.model_structure.parameters.rho is not None:
                    days_since = np.exp(-(1 / self.model_structure.parameters.rho.values[0]) * (1 + self.premises_status_days[
                        N_a[self.premises_status[day, N_a] == 4]]))[:, np.newaxis]
                    p_ij[self.premises_status[day, N_a] == 4, :] = 1 - np.exp(
                        -sim_susceptibility * self.gamma[self.premises_status[day, N_a[self.premises_status[day, N_a] == 4]] - 2, np.newaxis] *
                        new_transmissibility[N_a[self.premises_status[day, N_a] == 4], np.newaxis] * days_since * season_time *
                        K_ij[self.premises_status[day, N_a] == 4])
                p_aj = 1 - np.prod(1 - p_ij, axis=0)
                exp_mask = N_b[np.random.rand(n_b) < p_aj]
                expose_event[exp_mask] = True
        self.premises_status_days[np.where((self.premises_status[day, :] > 0) & (self.premises_status[day, :] <= 4))[0]] += 1
        self.premises_status[day + 1, :] = self.premises_status[day, :]
        self.premises_status[day + 1, (self.premises_status[day, :] == 0) & expose_event] = 1
        self.premises_status[day + 1, other_events == 1] += 1
        self.premises_status[day + 1, other_events == 2] += 2
        self.premises_status_days[other_events > 0] = 0
        self.premises_status_days[(self.premises_status[day, :] == 0) & expose_event] = 1
        self.infected_premises[rep] = np.append(self.infected_premises[rep], np.where((self.premises_status[day, :] == 0) & expose_event)[0])
        self.exposure_day[rep] = np.append(self.exposure_day[rep], np.full(np.sum((self.premises_status[day, :] == 0) & expose_event), day))

    def get_initial_conditions(self, rep):
        if self.initial_condition_type == 0:
            data_exposed = self.model_structure.data.report_day - (sum(x for x in self.model_structure.transitions[:(self.model_structure.data_compartments_idx[0] - 1)] if x is not None) + np.round(self.non_fixed_transitions[self.model_structure.data.infected_premises]).astype(int))
            initial_premises_idx = np.where((data_exposed <= 0) & (self.model_structure.data.report_day > -self.model_structure.transitions[2]))[0]
            initial_premises = self.model_structure.data.infected_premises[initial_premises_idx]
            self.infected_premises[rep] = copy.deepcopy(initial_premises)
            self.exposure_day[rep] = data_exposed[initial_premises_idx]
            initial_type = np.zeros(len(initial_premises), dtype=int)
            initial_type[np.isin(initial_premises, self.model_structure.data.infected_premises[np.where((data_exposed <= -self.model_structure.transitions[0]) & (self.model_structure.data.report_day > 0))[0]])] = 1
            initial_type[np.isin(initial_premises, self.model_structure.data.infected_premises[np.where((self.model_structure.data.report_day <= 0) & (self.model_structure.data.report_day > -self.model_structure.transitions[2]))[0]])] = 2
            initial_time = np.zeros(len(initial_premises), dtype=int)
            initial_time[initial_type == 0] = -data_exposed[initial_premises_idx][initial_type == 0]
            initial_time[initial_type == 1] = -data_exposed[initial_premises_idx][initial_type == 1] - self.model_structure.transitions[0]
            initial_time[initial_type == 2] = -self.model_structure.data.report_day[initial_premises_idx][initial_type == 2]
            self.premises_status_days[initial_premises] = initial_time
            self.premises_status[0, initial_premises] = initial_type + 1
        else:
            raise ValueError("Only initial_condition_type 0 (initial infected from posterior) is currently implemented.")

    def save_projections(self, dir='../outputs/'):
        """Save the projections to files."""
        if self.sellke:
            sellke_string = '_sellke'
        else:
            sellke_string = ''
        rep_string = f'_reps{self.reps}'
        np.save(f'{dir}simulation_{self.model_structure.chain_string}{sellke_string}{rep_string}_report_day.npy', self.report_day_projections)
        np.save(f'{dir}simulation_{self.model_structure.chain_string}{sellke_string}{rep_string}_report_premises.npy', self.report_premises_projections)
        np.save(f'{dir}simulation_{self.model_structure.chain_string}{sellke_string}{rep_string}_report_rep.npy', self.report_rep_projections.astype(int))
        np.save(f'{dir}simulation_{self.model_structure.chain_string}{sellke_string}{rep_string}_report_time.npy', self.report_time_projections)

    def load_projections(self, dir='../outputs/'):
        """Load projections."""
        if self.sellke:
            sellke_string = '_sellke'
        else:
            sellke_string = ''
        rep_string = f'_reps{self.reps}'
        self.report_day_projections = np.load(f'{dir}simulation_{self.model_structure.chain_string}{sellke_string}{rep_string}_report_day.npy')
        self.report_premises_projections = np.load(f'{dir}simulation_{self.model_structure.chain_string}{sellke_string}{rep_string}_report_premises.npy')
        self.report_rep_projections = np.load(f'{dir}simulation_{self.model_structure.chain_string}{sellke_string}{rep_string}_report_rep.npy')
        self.report_time_projections = np.load(f'{dir}simulation_{self.model_structure.chain_string}{sellke_string}{rep_string}_report_time.npy')

class Plotting:
    def __init__(self, model_fitting=None, model_simulator=None):
        if model_fitting is not None:
            self.model_fitting = model_fitting
            self.model_structure = model_fitting.model_structure
        if model_simulator is not None:
            self.model_simulator = model_simulator
            self.model_structure = model_simulator.model_structure

    def plot_parameter_chains(self):
        """Plot MCMC chains for parameters."""
        if self.model_fitting is None:
            raise ValueError("model_fitting is required to plot parameter chains.")
        n_parameters = self.model_fitting.parameter_posterior.shape[1]
        fig, ax = plt.subplots(np.ceil(np.sqrt(n_parameters)).astype(int), np.ceil(n_parameters/np.ceil(np.sqrt(n_parameters))).astype(int), figsize=(10, 10))
        i = 0
        for par_name, par in self.model_structure.parameters.fitted_parameters().items():
            for j in range(np.sum(par.fitted)):
                row = i // ax.shape[1]
                col = i % ax.shape[1]
                ax[row, col].plot(self.model_fitting.parameter_chains[:, i, :].T, label=f'{par_name}_{j}')
                ax[row, col].set_title(f'$\\{par_name}_{j}$')
                i += 1
        plt.tight_layout()

    def plot_parameter_posteriors(self):
        """Plot posterior distributions for parameters."""
        if self.model_fitting is None:
            raise ValueError("model_fitting is required to plot parameter posteriors.")
        n_parameters = self.model_fitting.parameter_posterior.shape[1]
        fig, ax = plt.subplots(np.ceil(np.sqrt(n_parameters)).astype(int), np.ceil(n_parameters/np.ceil(np.sqrt(n_parameters))).astype(int), figsize=(10, 10))
        i = 0
        for par_name, par in self.model_structure.parameters.fitted_parameters().items():
            for j in range(np.sum(par.fitted)):
                row = i // ax.shape[1]
                col = i % ax.shape[1]
                ax[row, col].hist(self.model_fitting.parameter_posterior[:, i], bins=50, density=True, alpha=0.7)
                ax[row, col].set_title(f'$\\{par_name}_{j}$')
                i += 1
        plt.tight_layout()
        plt.show()

    def plot_projections(self):
        """Plot simulation projections of weekly notified premises."""
        if self.model_simulator is None:
            raise ValueError("model_simulator is required to plot projections.")
        plt.figure(figsize=(10, 6))
        weeks = np.ceil(self.model_simulator.report_day_projections / 7).astype(int) - 1
        n_weeks = np.max(weeks + 1)
        n_reps = len(np.unique(self.model_simulator.report_rep_projections))
        ips_array = np.zeros((n_reps, n_weeks), dtype=int)
        np.add.at(ips_array, (self.model_simulator.report_rep_projections.astype(int), weeks), 1)

        mean_ips = np.mean(ips_array, axis=0)
        lower_ci = np.percentile(ips_array, 2.5, axis=0)
        upper_ci = np.percentile(ips_array, 97.5, axis=0)

        plt.fill_between(range(n_weeks), lower_ci, upper_ci, color='lightblue', alpha=0.5, label='95% Credible Interval')
        plt.plot(range(n_weeks), mean_ips, color='blue', label='Mean Weekly IPs')

        plt.xlabel('Report Day')
        plt.ylabel('Weekly IPs')
        plt.show()

# @njit
def cauchy_kernel(distance2, delta, omega):
    """Numba-compiled Cauchy kernel calculation."""
    base = delta * delta + distance2
    return delta * (base ** -omega)

# @njit
def exp_kernel(distance2, delta):
    """Numba-compiled Exponential kernel calculation."""
    return np.exp(-np.sqrt(distance2) * delta)