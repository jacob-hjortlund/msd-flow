import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from pathlib import Path
from astropy.io import fits
from scipy.ndimage import zoom
from matplotlib.colors import LogNorm
from astropy.cosmology import Planck15
from astropy.nddata import block_reduce

# ---------------------------------------------------------
# 1. Load the FITS file and extract data/metadata
# ---------------------------------------------------------
DATA_DIR = Path("/home/jacob/PhD/Projects/msd-flow/data")
fits_files = sorted(list(DATA_DIR.glob("*.fits")))[1:]
real_fits = fits_files[0]
ideal_fits = fits_files[1]

with fits.open(real_fits) as hdul:
    # Assuming the image is in the primary HDU (index 0)
    realistic_image_nanomaggies = hdul[8].data
    realistic_header = hdul[8].header
    
with fits.open(ideal_fits) as hdul:
    # Assuming the image is in the primary HDU (index 0)
    idealized_image = hdul[4].data
    idealized_header = hdul[4].header

fov = idealized_header['FOVSIZE'] # kpc
physical_scale = idealized_header['CDELT1'] # kpc / pixel
z = idealized_header['REDSHIFT']

# Define our cosmology
cosmo = Planck15

# Calculate Angular Diameter Distance in parsecs
D_A = cosmo.angular_diameter_distance(z).to(u.kpc).value

# Calculate the old angular pixel scale (from 100 pc physical scale)
arcsec_per_radian = u.rad.to(u.arcsec)
theta_old = (physical_scale / D_A) * arcsec_per_radian 

# Define target telescope scale
theta_new = 0.168 # arcsec/pixel
zoom_factor = theta_old / theta_new

print(f"Original angular scale: {theta_old:.4f} arcsec/pixel")
print(f"Target angular scale:   {theta_new:.4f} arcsec/pixel")
print(f"Zoom factor:            {zoom_factor:.4f}")

valid_mask = idealized_image < 99.0

# 2. Calculate the physical angular area of one original pixel
area_old = theta_old**2 # in arcsec^2

# 3. Convert surface brightness to total linear flux per pixel
# Formula: Flux = Area * 10^(-0.4 * magnitude)
idealized_flux_image = np.zeros_like(idealized_image)
idealized_flux_image[valid_mask] = area_old * (10**(-0.4 * idealized_image[valid_mask]))
idealized_image_nanomaggies = idealized_flux_image * 1e9 / area_old

# 1. Find the maximum integer factor we can safely compress by
integer_factor = int(1 / zoom_factor)  # For 0.1075, this is 9

# 2. Block reduce by SUMMING. 
# A 9x9 grid of pixels is summed into 1 pixel. Flux is perfectly conserved.
intermediate_flux_image = block_reduce(idealized_flux_image, integer_factor, func=np.sum)

# 3. Calculate the remaining fractional zoom needed
# We already shrunk by 9, so we only need to shrink by the remainder
fractional_zoom = zoom_factor * integer_factor # e.g., 0.1075 * 9 = 0.9675

# 4. Interpolate the final small step 
telescope_flux_image = zoom(intermediate_flux_image, fractional_zoom, order=1)

# 5. Apply the surface brightness/flux scaling for the fractional step
telescope_flux_image = telescope_flux_image / (fractional_zoom**2)

# 6. Global Flux Correction
# Force exact numeric conservation to fix floating-point drift from the fractional zoom
flux_ratio = np.sum(idealized_flux_image) / np.sum(telescope_flux_image)
telescope_flux_image = telescope_flux_image * flux_ratio

dimming_factor = 1#(1 + z)**-5
telescope_flux_image = telescope_flux_image * dimming_factor

from astropy.convolution import Gaussian2DKernel, convolve_fft

# 1. Define your telescope's seeing and pixel scale
seeing_fwhm_arcsec = 0.6  # Typical ground-based seeing (e.g., Subaru HSC)
pixel_scale_arcsec = theta_new  # Your target scale (0.15 arcsec/pix)

# 2. Convert FWHM in arcseconds to standard deviation (sigma) in pixels
# The mathematical relationship for a Gaussian is FWHM = 2.355 * sigma
fwhm_pixels = seeing_fwhm_arcsec / pixel_scale_arcsec
sigma_pixels = fwhm_pixels / 2.355

# 3. Generate the 2D Gaussian PSF kernel
psf_kernel = Gaussian2DKernel(x_stddev=sigma_pixels)

# 4. Convolve the linear flux image with the PSF
# We use convolve_fft because it is vastly faster than standard convolution for images
telescope_flux_image = convolve_fft(telescope_flux_image, psf_kernel, normalize_kernel=True)

sb_limit_mag = 28.5
sb_limit_area_arcsec = 10.0 * 10.0 # 100 arcsec^2 box
sigma_level = 5.0 # The 28.5 mag is a 5-sigma limit

# 2. Find the 1-sigma flux noise over that entire 100 arcsec^2 box
flux_limit_box = sb_limit_area_arcsec * (10**(-0.4 * sb_limit_mag))
one_sigma_flux_box = flux_limit_box / sigma_level

# 3. Scale the noise down to a single pixel using Poisson/Gaussian statistics
pixel_area_arcsec = theta_new**2 
area_ratio = pixel_area_arcsec / sb_limit_area_arcsec

# The noise scales with the square root of the area ratio!
noise_std_flux = one_sigma_flux_box * np.sqrt(area_ratio)

# Add Gaussian noise to the flux array
noise_array = np.random.normal(loc=0.0, scale=noise_std_flux, size=telescope_flux_image.shape)
telescope_flux_image = telescope_flux_image + noise_array

print(f"Original image shape: {idealized_flux_image.shape}")
print(f"New image shape: {telescope_flux_image.shape}")
print(f"Original Total Flux: {np.sum(idealized_flux_image):.4g}")
print(f"Resampled Total Flux: {np.sum(telescope_flux_image):.4g}")


telescope_image_nanomaggies = telescope_flux_image * 1e9

vmin_ideal = np.min(idealized_image_nanomaggies)
vmax_ideal = np.nanpercentile(idealized_image_nanomaggies, 99.95)

vmin_real = np.min(realistic_image_nanomaggies)
vmax_real = np.nanpercentile(realistic_image_nanomaggies, 99.95)

vmin_sim = np.min(telescope_image_nanomaggies)
vmax_sim = np.nanpercentile(telescope_image_nanomaggies, 99.95)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot Original Idealized Image
im1 = axes[0].imshow(
    idealized_image_nanomaggies, origin='lower', cmap='magma', vmin=vmin_ideal, vmax=vmax_ideal
)
axes[0].set_title("Idealized Image")
axes[0].set_title(f"Idealized Physical Image\n(100 pc/pix, {idealized_image_nanomaggies.shape[0]}x{idealized_image_nanomaggies.shape[1]})")
axes[0].set_xlabel("Pixels")
axes[0].set_ylabel("Pixels")
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="nanomaggie")

im2 = axes[1].imshow(
    realistic_image_nanomaggies, origin='lower', cmap='magma', vmin=vmin_real, vmax=vmax_real, #interpolation='nearest'
)
axes[1].set_title(f"HSC-Realistic Image\n({theta_new:.3f} arcsec/pix, {realistic_image_nanomaggies.shape[0]}x{realistic_image_nanomaggies.shape[1]})")
axes[1].set_xlabel("Pixels")
axes[1].set_ylabel("Pixels")
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="nanomaggie")


im3 = axes[2].imshow(
    telescope_image_nanomaggies, origin='lower', cmap='magma', vmin=vmin_sim, vmax=vmax_sim, #interpolation='nearest'
)
axes[2].set_title(f"Mock Telescope Image with Noise & PSF\n({theta_new:.3f} arcsec/pix, {telescope_image_nanomaggies.shape[0]}x{telescope_image_nanomaggies.shape[1]})")
axes[2].set_xlabel("Pixels")
axes[2].set_ylabel("Pixels")

fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="nanomaggie")
fig.tight_layout()