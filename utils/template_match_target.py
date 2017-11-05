##########################
#template match functions#
########################################################################

import numpy as np
from skimage.feature import match_template
import cv2

def template_match_target(target, minrad=3, maxrad=50, match_thresh2=50, template_thresh=0.6, target_thresh=0.1):
    #HYPERPARAMETERS
    # MATCH_THRESH2: for template matching, if (x1-x2)^2 + (y1-y2)^2 + (r1-r2)^2 < match_thresh2, remove (x2,y2,r2) circle (it is a duplicate). For predicted target -> csv matching (i.e. in template_match_target_to_csv), if (x1-x2)^2 + (y1-y2)^2 + (r1-r2)^2 < match_thresh2, positive match. Maybe these should technically be separate parameters, but to first order they should be the same...
    # TEMPLATE_THRESH: 0-1 range, if scikit-image's template matching probability > template_thresh, count as detection
    # TARGET_THRESH: 0-1 range, set pixel values > target_thresh to 1, and pixel values < target_thresh -> 0
    # minrad - keep in mind that if the predicted target has thick rings, a small ring of diameter ~ ring_thickness could be detected by match_filter.
    
    # minrad/maxrad are the radii to search over during template matching
    # hyperparameters, probably don't need to change
    ring_thickness = 2       #thickness of rings for the templates. 2 seems to work well.
    
    # target - can be predicted or ground truth
    target[target >= target_thresh] = 1
    target[target < target_thresh] = 0
    
    #radii = np.linspace(minrad,maxrad,maxrad-minrad,dtype=int)
    radii = np.arange(minrad,maxrad+1,1,dtype=int)
    coords = []     #coordinates extracted from template matching
    corr = []       #correlation coefficient for coordinates set
    for r in radii:
        # template
        n = 2*(r+ring_thickness+1)
        template = np.zeros((n,n))
        cv2.circle(template, (r+ring_thickness+1,r+ring_thickness+1), r, 1, ring_thickness)
        
        # template match - result is nxn array of probabilities
        result = match_template(target, template, pad_input=True)   #skimage
        index_r = np.where(result > template_thresh)
        coords_r = np.asarray(zip(*index_r))
        corr_r = np.asarray(result[index_r])
        
        # store x,y,r
        for c in coords_r:
            coords.append([c[1],c[0],r])
        for l in corr_r:
            corr.append(np.abs(l))

    # remove duplicates from template matching at neighboring radii/locations
    coords, corr = np.asarray(coords), np.asarray(corr)
    i, N = 0, len(coords)
    while i < N:
        diff = (coords - coords[i])**2
        diffsum = np.asarray([sum(x) for x in diff])
        index = diffsum < match_thresh2
        if len(np.where(index==True)[0]) > 1:
            #replace current coord with match_template'd max-correlation coord from duplicate list
            coords_i, corr_i = coords[np.where(index==True)], corr[np.where(index==True)]
            coords[i] = coords_i[corr_i == np.max(corr_i)][0]
            index[i] = False
            coords = coords[np.where(index==False)]
        N, i = len(coords), i+1

    # This might not be necessary if minrad > ring_thickness, but probably good to keep as a failsafe
    # remove small false craters that arise because of thick edges
    i, N = 0, len(coords)
    dim = target.shape[0]
    while i < N:
        x,y,r = coords[i]
        if r < 6:   #this effect is not present for large craters
            mask = np.zeros((dim,dim))
            cv2.circle(mask, (x,y), int(np.round(r)), 1, thickness=-1)
            crater = target[mask==1]
            if np.sum(crater) == len(crater):   #crater is completely filled in, likely a false positive
                coords = np.delete(coords, i, axis=0)
                N = len(coords)
            else:
                i += 1
        else:
            i += 1

    return coords


def template_match_target_to_csv(target, csv, minrad=3, maxrad=50, match_thresh2=50, template_thresh=0.6, target_thresh=0.1):

    #get coordinates from template matching
    templ_coords = template_match_target(target, minrad, maxrad, match_thresh2, template_thresh, target_thresh)

    #find max detected crater radius
    maxr = 0
    if len(templ_coords > 0):
        x,y,r = templ_coords.T
        maxr = np.max(r)
    
    #If remove_large_craters_csv == 1, see how recall improves when large craters are excluded.
    remove_large_craters_csv = 0
    if remove_large_craters_csv == 1:
        index = np.where((csv.T[2] < maxr)&(csv.T[2] > minrad))
        if len(index[0]) > 0:
            csv_coords = csv[index]
        else:
            print "all craters are larger than max detected radius"
            csv_coords = csv
    else:
        csv_coords = csv

    # compare template-matched results to "ground truth" csv input data
    N_match = 0
    csv_duplicate_flag = 0
    N_csv, N_templ = len(csv_coords), len(templ_coords)
    for tc in templ_coords:
        #diff = (csv_coords - tc)**2
        #diffsum = np.asarray([sum(x) for x in diff])
        diffsum = np.sum((csv_coords - tc)**2,axis=1)
        index = diffsum > match_thresh2
        N = len(np.where(index==False)[0])
        if N > 1:
            csv_duplicate_flag = 1  #more than one match found
        N_match += N
        csv_coords = csv_coords[index]
        if len(csv_coords) == 0:
            break

    return N_match, N_csv, N_templ, maxr, csv_duplicate_flag


