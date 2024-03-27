import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp2d


def sample_random_subgrids(RR_pixels,ZZ_pixels,nr=64,nz=64):
  delta_r_min = .33*(RR_pixels.max()-RR_pixels.min())
  delta_r_max = .75*(RR_pixels.max()-RR_pixels.min())

  delta_z_min = .2*(ZZ_pixels.max()-ZZ_pixels.min())
  delta_z_max = .75*(ZZ_pixels.max()-ZZ_pixels.min())

  delta_r = np.random.uniform(delta_r_min,delta_r_max,1)
  r0 = np.random.uniform(RR_pixels.min(),RR_pixels.min()+delta_r_max-delta_r,1)

  delta_z = np.random.uniform(delta_z_min,delta_z_max,1)
  z0 = np.random.uniform(ZZ_pixels.min(),ZZ_pixels.min()+delta_z_max-delta_z,1)

  rr = np.linspace(r0,r0+delta_r,nr)
  zz = np.linspace(z0,z0+delta_z,nz)

  rr_grid, zz_grid = np.meshgrid(rr,zz,indexing='xy')

  return rr_grid, zz_grid


def get_box_from_grid(rr_grid, zz_grid):
  return np.array([
      [rr_grid.min(), zz_grid.min()],
      [rr_grid.max(), zz_grid.min()],
      [rr_grid.max(), zz_grid.max()],
      [rr_grid.min(), zz_grid.max()],
      [rr_grid.min(), zz_grid.min()]])


def get_grid_from_box(box,nr=64,nz=64):
  rr = np.linspace(box[:,0].min(),box[:,0].max(),nr)
  zz = np.linspace(box[:,1].min(),box[:,1].max(),nz)
  rr_grid, zz_grid = np.meshgrid(rr,zz,indexing='xy')
  return rr_grid, zz_grid


# def interp_fun(f,RR_pixels,ZZ_pixels,rr_grid,zz_grid,kind):
#   x_pts = RR_pixels[0,:].ravel()
#   y_pts = ZZ_pixels[:,0].ravel()
#   # interp_func = RegularGridInterpolator((x_pts, y_pts), f.T,kind)
#   interp_func = interp2d((x_pts, y_pts), f.T,kind)
#   f_int = interp_func(np.column_stack((
#         rr_grid.reshape(-1, 1),
#         zz_grid.reshape(-1, 1),
#         )),method='quintic').reshape(rr_grid.shape)
#   return f_int

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def interp_fun(f,RR_pixels,ZZ_pixels,rr_grid,zz_grid,kind='quintic'):
  x_pts = RR_pixels[0,:].ravel()
  y_pts = ZZ_pixels[:,0].ravel()
  # interp_func = RegularGridInterpolator((x_pts, y_pts), f.T,kind)
  interp_func = interp2d(x_pts, y_pts, f,kind=kind)
  f_int = interp_func(rr_grid[0,:].ravel(),zz_grid[:,0].ravel()).reshape(rr_grid.shape)
  return f_int


def def_grids_and_interp(f,rhs,RR_pixels,ZZ_pixels,nr=64,nz=64,kind='quintic'):
  rr_grid, zz_grid = sample_random_subgrids(RR_pixels,ZZ_pixels,nr,nz)
  f_grid = interp_fun(f,RR_pixels,ZZ_pixels,rr_grid, zz_grid,kind=kind)
  rhs_grid = interp_fun(rhs,RR_pixels,ZZ_pixels,rr_grid, zz_grid,kind=kind)
  return rr_grid, zz_grid, f_grid, rhs_grid