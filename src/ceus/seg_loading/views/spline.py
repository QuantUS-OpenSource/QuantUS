import numpy as np
import pyvista as pv
import scipy.interpolate as interpolate
from numpy.matlib import repmat


def removeDuplicates(ar):
    # Credit: https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
    seen = set()
    seenAdd = seen.add
    return [x for x in ar if not (tuple(x) in seen or seenAdd(tuple(x)))]

def calculateSpline(xpts, ypts, zpts=None):  # 2D spline interpolation
    cv = []
    for i in range(len(xpts)):
        if zpts is not None:
            cv.append([xpts[i], ypts[i], zpts[i]])
        else:
            cv.append([xpts[i], ypts[i]])
    
    # Remove duplicate points which cause "ValueError: Invalid inputs" in splprep
    cv = np.array(cv)
    if len(cv) > 1:
        # Calculate distances between consecutive points
        diffs = np.diff(cv, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        # Keep first point and points that are sufficiently far from the previous one
        mask = np.concatenate(([True], dists > 1e-5))
        cv = cv[mask]

    if len(cv) < 2:
        if zpts is not None:
            return np.array([cv[0][0]]), np.array([cv[0][1]]), np.array([cv[0][2]])
        return np.array([cv[0][0]]), np.array([cv[0][1]])

    if len(cv) == 2:
        tck, _ = interpolate.splprep(cv.T, s=0.0, k=1)
    elif len(cv) == 3:
        tck, _ = interpolate.splprep(cv.T, s=0.0, k=2)
    else:
        tck, _ = interpolate.splprep(cv.T, s=0.0, k=3)

    if zpts is not None:
        x, y, z = np.array(interpolate.splev(np.linspace(0, 1, 100), tck))
        return x, y, z
    x, y = np.array(interpolate.splev(np.linspace(0, 1, 1000), tck))
    return x, y


def ellipsoidFitLS(pos):
    # centre coordinates on origin
    pos = pos - np.mean(pos, axis=0)

    # build our regression matrix
    A = pos**2

    # vector of ones
    Ones = np.ones(len(A))

    # least squares solver
    B, _, _, _ = np.linalg.lstsq(A, Ones, rcond=None)

    # solving for a, b, c
    a_ls = np.sqrt(1.0 / B[0])
    b_ls = np.sqrt(1.0 / B[1])
    c_ls = np.sqrt(1.0 / B[2])

    return (a_ls, b_ls, c_ls)


def calculateSpline3D(points):
    # If the points have a 4th dimension (time), we strip it as pyvista expects (x, y, z)
    points = np.array(points)
    if points.shape[1] == 4:
        points = points[:, :3]
    
    cloud = pv.PolyData(points, force_float=False)
    volume = cloud.delaunay_3d(alpha=100)
    shell = volume.extract_geometry() # type: ignore
    final = shell.triangulate() # type: ignore
    final.smooth(n_iter=1000) # type: ignore
    faces = final.faces.reshape((-1, 4)) # type: ignore
    faces = faces[:, 1:]
    arr = final.points[faces] # type: ignore

    arr = np.array(arr)

    output = set()
    for tri in arr:
        slope_2 = tri[2] - tri[1]
        start_2 = tri[1]
        slope_3 = tri[0] - tri[1]
        start_3 = tri[1]
        for i in range(100, -1, -1):
            bound_one = start_2 + ((i / 100) * slope_2)
            bound_two = start_3 + ((i / 100) * slope_3)
            cur_slope = bound_one - bound_two
            cur_start = bound_two
            for j in range(100, -1, -1):
                cur_pos = cur_start + ((j / 100) * cur_slope)
                output.add((int(cur_pos[0]), int(cur_pos[1]), int(cur_pos[2])))

    return output

def iqToRf(iqData, rxFrequency, decimationFactor, carrierFrequency):
    import scipy.signal as ssg    
    iqData = ssg.resample_poly(iqData, decimationFactor, 1) # up-sample by decimation factor
    rfData = np.zeros(iqData.shape)
    t = [i*(1/rxFrequency) for i in range(iqData.shape[0])]
    for i in range(iqData.shape[1]):
        rfData[:,i] = np.real(np.multiply(iqData[:,i], np.exp(1j*(2*np.pi*carrierFrequency*np.transpose(t)))))
    return rfData