import numpy as np
import xarray as xr

## All of the files are from the simulation output except "MASK_FLUX" which corresponds to the mask where the basins are defined
def FreshWaterBudget(GRID_T,GRID_U,GRID_V,MESHMASK,MASK_FLUX):
    ## Global Mask (each value corresponds to a basin): 
    MK = MASK_FLUX.Bathymetry.values

    ## Number of basins: 
    N_BASINS = np.unique(MK)

    ## Number of vertical layers:
    NZ = GRID_T.deptht.shape[0]

    ## Multiply the mask for each vertical layer:
    MK_Z = MK * np.ones((NZ, MK.shape[0], MK.shape[1]))

    ## Concatenate all the mask (0/1 values):
    Gmask = np.zeros((N_BASINS.shape[0], MK_Z.shape[0], MK_Z.shape[1], MK_Z.shape[2]))

    j = 0
    for i in N_BASINS:
        MK_basin = np.zeros((MK_Z.shape[0], MK_Z.shape[1], MK_Z.shape[2]))
        MK_basin[MK_Z == i] = 1
        Gmask[j,:,:,:] = MK_basin
        j+=1

    ## Cell area @ T:
    CellArea = MESHMASK.e1t.values [0,:,:] * MESHMASK.e2t.values [0,:,:]
    ## Cell volume @ T:
    CellVol = (MESHMASK.e2t.values[0,:,:]* MESHMASK.e1t.values[0,:,:]* MESHMASK.e3t_0.values[0,:,:,:])

    ############################# Flux between basins : (Maffre. P, 2019)

    ## Parameters:
    NBASIN = N_BASINS.shape[0]
    periodic_x = False
    u_flx, v_flx = True, True

    ## Horizontal velocities fields:
    u_var = GRID_U.uocetr_eff
    v_var = GRID_V.vocetr_eff

    ## Annual mean: 
    u_field = np.nanmean(u_var, axis = 0)
    v_field = np.nanmean(v_var, axis = 0)

    ## Overlap:
    ix0 = 1
    ix1 = u_field.shape[2]-1
    x_overlap = 3

    # + + + + + + + + + + + + + + + + + + + + + + + + + + + + #
    # Matrix of fluxes: X_mat[i,j] = flux FROM box i TO box j #
    # + + + + + + + + + + + + + + + + + + + + + + + + + + + + #
    X_mat = np.zeros((NZ,NBASIN,NBASIN), dtype=float)

    for i in range(NBASIN):
        for j in list(range(i))+list(range(i+1,NBASIN)):
            for k in range(NZ):
                uij, vij = (), ()
                msk_uij = np.logical_and(Gmask[i,k,:,ix0:ix1-1], Gmask[j,k,:,ix0+1:ix1])
                msk_uij = np.logical_and(msk_uij, ~np.isnan(u_field[k,:,ix0:ix1-1]))


                if msk_uij.any():
                    uij = u_field[k,:,ix0:ix1-1][msk_uij]
                    pos = (uij >= 0)
                    X_mat[k,i,j] += np.nansum(uij[pos])
                    X_mat[k,j,i] -= np.nansum(uij[~pos])

                # v-fluxes
                msk_vij = np.logical_and(Gmask[i,k,:-1,ix0:ix1], Gmask[j,k,1:,ix0:ix1])
                msk_vij = np.logical_and(msk_vij, ~np.isnan(v_field[k,:-1,ix0:ix1]))

                if msk_vij.any():
                    vij = v_field[k,:-1,ix0:ix1][msk_vij]
                    pos = (vij > 0)
                    X_mat[k,i,j] += np.nansum(vij[pos])
                    X_mat[k,j,i] -= np.nansum(vij[~pos])


    # Flux computation [m^3/s]:
    Flux = np.zeros((NZ,NBASIN,NBASIN)) # (Z,basin_ref, surrounding_basins) positive values for water fluxes input in the basin_ref and reversly
    for i in range (NBASIN) : 
        Flux[:,i,:] = np.subtract(X_mat [:, :, i],X_mat [:, i, :])

    ############################# Averaged salinity & temperature of each basin in the upper 100 meters (it is possible to adjust this depth by changing "Z_max"):

    Z_max = 100
    ix_Z_sel = np.where(GRID_T.deptht <= Z_max)[0]

    ## Mask to handle the longitudinal overlap of the ORCA2 grid:
    MK_ovlap = MESHMASK.tmaskutil[0,:,:].values * np.ones((NZ, MESHMASK.tmaskutil.shape[1], MESHMASK.tmaskutil.shape[2]))

    ## Gmask -> Gmask_Nan
    Gmask_Nan = Gmask.copy()
    Gmask_Nan[Gmask_Nan == 0] = np.nan

    ## Ocean mask:
    MK_Ocean = GRID_T.so.mean(axis = 0).values.copy()
    MK_Ocean[np.isnan(MK_Ocean) == False] = 1

    ## Compute the averaged salinity:
    S = GRID_T.so.mean(axis = 0) # Annual mean global salinity
    SMean_100m = np.zeros((NBASIN))
    for i in range (NBASIN):
        SMean_100m[i] = np.nansum(S[ix_Z_sel,:,:] * CellVol[ix_Z_sel,:,:] * Gmask_Nan[i,ix_Z_sel,:,:] * MK_ovlap[ix_Z_sel,:,:])/np.nansum(CellVol[ix_Z_sel,:,:] * Gmask_Nan[i,ix_Z_sel,:,:] * MK_ovlap[ix_Z_sel,:,:] * MK_Ocean[ix_Z_sel,:,:])

    ## Compute the averaged temperature:
    T = GRID_T.thetao.mean(axis = 0) # Annual mean global salinity
    TMean_100m = np.zeros((NBASIN))
    for i in range (NBASIN):
        TMean_100m[i] = np.nansum(T[ix_Z_sel,:,:] * CellVol[ix_Z_sel,:,:] * Gmask_Nan[i,ix_Z_sel,:,:] * MK_ovlap[ix_Z_sel,:,:])/np.nansum(CellVol[ix_Z_sel,:,:] * Gmask_Nan[i,ix_Z_sel,:,:] * MK_ovlap[ix_Z_sel,:,:] * MK_Ocean[ix_Z_sel,:,:])

    ############################# Hydrological cycle:
    # A = Total freshwater input (Sv) (Runoff + Precipitation - Evaporation + sea ice melting (snow over sea ice + sublimation over sea ice - water flux due to freezing/melting + snow over ice free ocean))
    FWB = ((np.nanmean(GRID_T.friver, axis = 0) + np.nanmean(GRID_T.rain - GRID_T.evap_ao_cea, axis = 0) + np.nanmean(GRID_T.subl_ai_cea, axis = 0)  + np.nanmean(GRID_T.snow_ai_cea, axis = 0)  - np.nanmean(GRID_T.fmmflx, axis = 0)  + np.nanmean(GRID_T.snow_ao_cea, axis = 0)) * 10 ** -9)\
    * CellArea * MESHMASK.tmaskutil[0,:,:]
    Precip = np.nanmean(GRID_T.rain, axis = 0) * 10 ** -9 * CellArea * MESHMASK.tmaskutil[0,:,:]
    Evap = np.nanmean(GRID_T.evap_ao_cea, axis = 0) * 10 ** -9 * CellArea * MESHMASK.tmaskutil[0,:,:]
    Runoff = np.nanmean(GRID_T.friver, axis = 0) * 10 ** -9 * CellArea * MESHMASK.tmaskutil[0,:,:]
    SeaIce = np.nanmean(GRID_T.subl_ai_cea, axis = 0)  + np.nanmean(GRID_T.snow_ai_cea, axis = 0)  - np.nanmean(GRID_T.fmmflx, axis = 0)  + np.nanmean(GRID_T.snow_ao_cea, axis = 0) * 10 ** -9\
    * MESHMASK.tmaskutil[0,:,:]

    # FWB for each basin (FWB total, Precip, Evap, Runoff, SeaIce):
    FWB_basin = np.zeros((NBASIN,5))

    for i in range (NBASIN):
        FWB_basin[i,0] = np.nansum(FWB * Gmask_Nan[i,0,:,:])
        FWB_basin[i,1] = np.nansum(Precip * Gmask_Nan[i,0,:,:])
        FWB_basin[i,2] = np.nansum(Evap * Gmask_Nan[i,0,:,:])
        FWB_basin[i,3] = np.nansum(Runoff * Gmask_Nan[i,0,:,:])
        FWB_basin[i,4] = np.nansum(SeaIce * Gmask_Nan[i,0,:,:])
    return Flux, SMean_100m, TMean_100m, FWB_basin
