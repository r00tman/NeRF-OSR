import importlib
import scipy
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2


class Rotation(object):

	def __init__(self):
		super(Rotation, self).__init__()
		rows_x = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8]
		cols_x = [0, 2, 1, 3, 7, 5, 6, 8, 4, 6, 8]

		data_x_p90 = [1,-1,1,1,-1,-1,-1/2,-np.sqrt(3)/2,1,-np.sqrt(3)/2,1/2]
		data_x_n90 = [1,1,-1,1,1,-1,-1/2,-np.sqrt(3)/2,-1,-np.sqrt(3)/2,1/2]

		self.Rot_X_p90 = scipy.sparse.coo_matrix((data_x_p90,(rows_x,cols_x)), shape=(9,9)).toarray()
		self.Rot_X_n90 = scipy.sparse.coo_matrix((data_x_n90,(rows_x,cols_x)), shape=(9,9)).toarray()

		self.rows_z = [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8]
		self.cols_z = [0, 1, 3, 2, 1, 3, 4, 8, 5, 7, 6, 5, 7, 4, 8]


	def rot_SH(self, SH, thetaX, thetaY, thetaZ):


		Rot_y = []
		Rot_x = []
		Rot_z = []
		# for rot in np.arange(0,2*np.pi,np.pi/18):
		for rotx, roty, rotz in zip(thetaX, thetaY, thetaZ):
			Rot_z.append(self.rot_z(rotz))
			Rot_y.append(self.rot_y(roty))
			Rot_x.append(self.rot_x(rotx))

		Rot_y = np.stack(Rot_y, axis=0)
		Rot_x = np.stack(Rot_x, axis=0)
		Rot_z = np.stack(Rot_z, axis=0)


		Rot = np.matmul(Rot_z, np.matmul(Rot_y, Rot_x))
		rot_SH = np.matmul(Rot, SH)

		return rot_SH


	def rot_z(self, thetaZ):
		data_Z = [1,np.cos(thetaZ),np.sin(thetaZ),1,-np.sin(thetaZ),np.cos(thetaZ),np.cos(2*thetaZ),np.sin(2*thetaZ),np.cos(thetaZ),np.sin(thetaZ),1,-np.sin(thetaZ),np.cos(thetaZ),-np.sin(2*thetaZ),np.cos(2*thetaZ)]

		return scipy.sparse.coo_matrix((data_Z,(self.rows_z, self.cols_z)), shape=(9,9)).toarray()

	def rot_y(self, thetaY):
		data_Z = [1,np.cos(thetaY),np.sin(thetaY),1,-np.sin(thetaY),np.cos(thetaY),np.cos(2*thetaY),np.sin(2*thetaY),np.cos(thetaY),np.sin(thetaY),1,-np.sin(thetaY),np.cos(thetaY),-np.sin(2*thetaY),np.cos(2*thetaY)]
		rotM_z = scipy.sparse.coo_matrix((data_Z,(self.rows_z, self.cols_z)), shape=(9,9)).toarray()

		return np.matmul(self.Rot_X_p90, np.matmul(rotM_z, self.Rot_X_n90))


	def rot_x(self, thetaX):
		data_Z = [1,np.cos(thetaX),np.sin(thetaX),1,-np.sin(thetaX),np.cos(thetaX),np.cos(2*thetaX),np.sin(2*thetaX),np.cos(thetaX),np.sin(thetaX),1,-np.sin(thetaX),np.cos(thetaX),-np.sin(2*thetaX),np.cos(2*thetaX)]
		rotM_z = scipy.sparse.coo_matrix((data_Z,(self.rows_z, self.cols_z)), shape=(9,9)).toarray()
		
		return np.matmul(self.rot_y(np.pi/2), np.matmul(rotM_z, self.rot_y(-np.pi/2)))



def render_sphere_nm(radius, num):
	# nm is a batch of normal maps
	nm = []

	for i in range(num):
		### sphere (projected on circular image just like angular map)
		# span the regular grid for computing azimuth and zenith angular map
		height = 2*radius
		width = 2*radius
		centre = radius
		h_grid, v_grid = np.meshgrid(np.arange(1.,2*radius+1), np.arange(1.,2*radius+1))
		# grids are (-radius, radius)
		h_grid -= centre
		# v_grid -= centre
		v_grid = centre - v_grid
		# scale range of h and v grid in (-1,1)
		h_grid /= radius
		v_grid /= radius

		# z_grid is linearly spread along theta/zenith in range (0,pi)
		dist_grid = np.sqrt(h_grid**2+v_grid**2)
		dist_grid[dist_grid>1] = np.nan
		theta_grid = dist_grid * np.pi
		z_grid = np.cos(theta_grid)

		rho_grid = np.arctan2(v_grid,h_grid)
		x_grid = np.sin(theta_grid)*np.cos(rho_grid)
		y_grid = np.sin(theta_grid)*np.sin(rho_grid)

		# concatenate normal map
		nm.append(np.stack([x_grid,y_grid,z_grid],axis=2))


	# construct batch
	nm = np.stack(nm,axis=0)



	return nm




def sh_recon(nm, lighting):
	width = nm.shape[1]

	x = nm[:,:,:,0]
	y = nm[:,:,:,1]
	z = nm[:,:,:,2]

	# convert light probe to angular map(evenly distributed front and back environment), find light directions by new angular map
	azi = np.arctan2(y, x)
	zen = np.arccos(z)

	c1 = 0.282095
	c2 = 0.488603
	c3 = 1.092548
	c4 = 0.315392
	c5 = 0.546274

	# domega = 4*np.pi**2/width**2 * sinc(zen)
	domega = np.ones_like(zen)

	sh_basis = np.stack([c1 * domega, c2*y * domega, c2*z * domega, c2*x * domega, c3*x*y * domega, c3*y*z * domega, c4*(3*z*z-1) * domega, c3*x*z * domega, c5*(x*x-y*y) * domega], axis=1)
	sh_basis = np.expand_dims(sh_basis, axis=-1)
	lighting_recon = np.expand_dims(np.expand_dims(lighting,axis=-2),axis=-2) * sh_basis
	lighting_recon = np.sum(lighting_recon,axis=1)

	return lighting_recon



# perform spherical harmonics projection based on Cartesian Coords SH basis
def SH_proj(func,coords,width):
	# func and coords have shape (npix, 3[rgb]/[xyz])
	c1 = 0.282095
	c2 = 0.488603
	c3 = 1.092548
	c4 = 0.315392
	c5 = 0.546274

	x = coords[:,0,np.newaxis]
	y = coords[:,1,np.newaxis]
	z = coords[:,2,np.newaxis]
	theta = np.arccos(z)
	domega = 4*np.pi**2/width**2 * sinc(theta)

	coeffs = []
	coeffs.append(np.sum(func * c1 * domega, axis=0))
	coeffs.append(np.sum(func * c2*y * domega, axis=0))
	coeffs.append(np.sum(func * c2*z * domega, axis=0))
	coeffs.append(np.sum(func * c2*x * domega, axis=0))
	coeffs.append(np.sum(func * c3*x*y * domega, axis=0))
	coeffs.append(np.sum(func * c3*y*z * domega, axis=0))
	coeffs.append(np.sum(func * c4*(3*z*z-1) * domega, axis=0))
	coeffs.append(np.sum(func * c3*x*z * domega, axis=0))
	coeffs.append(np.sum(func * c5*(x*x-y*y) * domega, axis=0))

	coeffs = np.stack(coeffs,axis=0)
	return coeffs


def sinc(x):
    """Supporting sinc function
    """
    output = np.sin(x)/x
    output[np.isnan(output)] = 1.
    return output



def angularMap_dirs_r2r(em):
	# Coordinates following Will's convention
	h,w,_ = em.shape
	centre_h = h/2.
	centre_w = w/2.
	h_grid = np.arange(h,0,-1)
	# h_grid = np.arange(1,h+1)
	w_grid = np.arange(1,w+1)
	w_grid, h_grid = np.meshgrid(w_grid, h_grid)
	# w_grid and h_grid stand for x and y in range (-1,1)
	w_grid = ((w_grid-centre_w)/centre_w)
	h_grid = ((h_grid-centre_h)/centre_h)
	dis = np.sqrt(w_grid**2+h_grid**2)
	lightMask = dis<=1. # flattened mask

	# convert light probe to angular map(evenly distributed front and back environment), find light directions by new angular map
	azi = np.arctan2(h_grid, w_grid).astype(np.float32)
	zen = (dis*np.pi).astype(np.float32)



	x = -np.sin(zen)*np.sin(azi)
	y = np.cos(zen) 
	z = -np.sin(zen)*np.cos(azi) 

	lightDirs = np.stack([x[lightMask],y[lightMask],z[lightMask]], axis=1)

	em_r = em[:,:,0]
	em_g = em[:,:,1]
	em_b = em[:,:,2]
	lightColors = np.stack([em_r[lightMask], em_g[lightMask], em_b[lightMask]], axis=1)

	return lightDirs, lightColors, lightMask



def angularMap_dirs(em):
	h,w,_ = em.shape
	centre_h = h/2.
	centre_w = w/2.
	h_grid = np.arange(h,0,-1)
	# h_grid = np.arange(1,h+1)
	w_grid = np.arange(1,w+1)
	w_grid, h_grid = np.meshgrid(w_grid, h_grid)
	# w_grid and h_grid stand for x and y in range (-1,1)
	w_grid = ((w_grid-centre_w)/centre_w)
	h_grid = ((h_grid-centre_h)/centre_h)
	dis = np.sqrt(w_grid**2+h_grid**2)
	lightMask = dis<=1. # flattened mask

	# convert light probe to angular map(evenly distributed front and back environment), find light directions by new angular map
	azi = np.arctan2(h_grid, w_grid).astype(np.float32)
	zen = (dis*np.pi).astype(np.float32)

	# x->right, y->up, z->outward
	x = np.sin(zen)*np.cos(azi)
	y = np.sin(zen)*np.sin(azi)
	z = np.cos(zen)

	lightDirs = np.stack([x[lightMask],y[lightMask],z[lightMask]], axis=1)

	em_r = em[:,:,0]
	em_g = em[:,:,1]
	em_b = em[:,:,2]
	lightColors = np.stack([em_r[lightMask], em_g[lightMask], em_b[lightMask]], axis=1)

	return lightDirs, lightColors, lightMask



def latlongMap_dirs_r2r(em):
	# Coordinates following Will's convention
	h,w,_ = em.shape

	cols, rows = np.meshgrid(np.arange(w), np.arange(h))

	theta = -np.pi*((2*cols)/w-1)
	phi = (rows*np.pi)/h



	x = -np.cos(phi)
	y = np.sin(phi)*np.cos(theta)
	z = np.sin(phi)*np.sin(theta)




	lightDirs = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)

	lightColors = em.reshape(-1, 3).astype(np.float32)

	return lightDirs, lightColors




def latlongMap_dirs(em):
	h,w,_ = em.shape
	azi, zen = np.meshgrid(np.arange(0,2*np.pi,2*np.pi/w), np.arange(np.pi/2,-np.pi/2,-np.pi/h))

	# x->right, y->up, z->inward
	x = -np.cos(zen)*np.sin(azi)
	y = np.sin(zen)
	z = np.cos(zen)*np.cos(azi)

	lightDirs = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)

	lightColors = em.reshape(-1, 3).astype(np.float32)

	return lightDirs, lightColors



def main():
	# path to the lat-long hdr image
	import ipdb; ipdb.set_trace()
	exr_img_paths = 'demo.exr'

	# read and resize the hdr image
	exr_img = imageio.imread(exr_img_paths, format='EXR-FI')
	exr_img = cv2.resize(exr_img, (360, 180), cv2.INTER_CUBIC)
	exr_h, exr_w = exr_img.shape[:2]


	# project the lat-long map to an angular map
	# define the direction for each pixel and interpolate the angular map by the pixels in lat-long map with similar direction
	exr_img_angMap = np.zeros((exr_h,exr_h,3), np.float32)
	latlongMap_dirs, latlongMap_values = latlongMap_dirs(exr_img)
	angularMap_dirs, _, angularMap_mask = angularMap_dirs(exr_img_angMap)
	# latlongMap_dirs, latlongMap_values = latlongMap_dirs_r2r(exr_img)
	# angularMap_dirs, _, angularMap_mask = angularMap_dirs_r2r(exr_img_angMap)

	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=1).fit(latlongMap_dirs)
	_, idxs = nbrs.kneighbors(angularMap_dirs)
	angularMap_values = latlongMap_values[idxs[:,0]]
	exr_img_angMap[angularMap_mask] = angularMap_values


	# project the angular hdr map to the spherical harmonics
	img_sh = SH_proj(angularMap_values, angularMap_dirs, exr_h)
	lighting_recon = sh_recon(np.float32(render_sphere_nm(100,1)), img_sh)
	lighting_validPix = lighting_recon[np.logical_not(np.isnan(lighting_recon))]
	lighting_recon = (lighting_recon - lighting_validPix.min()) / (lighting_validPix.max() - lighting_validPix.min())
	lighting_recon[np.isnan(lighting_recon)] = 0


	# rotate the original lighting by -pi/3 about x axis
	rotation = Rotation()
	rot = np.float32(np.dot(rotation.rot_y(0.), np.dot(rotation.rot_x(-np.pi/3), rotation.rot_z(0.))))
	rot_sh = np.matmul(rot, img_sh)[None]

	# recontruct the rotated lighting
	rot_lighting_recon = sh_recon(np.float32(render_sphere_nm(100,1)), rot_sh)
	rot_lighting_validPix = rot_lighting_recon[np.logical_not(np.isnan(rot_lighting_recon))]
	rot_lighting_recon = (rot_lighting_recon - rot_lighting_validPix.min()) / (rot_lighting_validPix.max() - rot_lighting_validPix.min())
	rot_lighting_recon[np.isnan(rot_lighting_recon)] = 0



	plt.figure(); plt.imshow(exr_img)
	plt.figure(); plt.imshow(exr_img_angMap)
	plt.figure(); plt.imshow(lighting_recon[0])
	plt.figure(); plt.imshow(rot_lighting_recon[0])
	plt.show()

if __name__ == '__main__':
	main()