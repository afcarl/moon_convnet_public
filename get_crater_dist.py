#The point of this script is to take the outputted numpy files generated from crater_distribution_extract_*.py and generate a list of unique craters, i.e. no duplicates. The key hyperparameters to tune are thresh_longlat2 and thresh_rad2, which is guided by comparing the unique distirbution to the ground truth (alanalldata.csv) data.
#First you need to generate predictions from crater_distribution_get_modelpreds.py

import numpy as np
import cPickle
from utils.template_match_target import *
from utils.rescale import *
import glob
from keras.models import load_model

#########################
def get_model_preds(CP):
    dim, nimgs = CP['dim'], CP['nimgs']

    data=np.load(CP['data_dir'])/255.
    data=data.reshape((len(data),dim,dim,1))
    data = data[:nimgs]
    if CP['rescale'] == 1:
        data = rescale(data)

    model = load_model(model_loc)
    preds = model.predict(data.astype('float32'))
    np.save(CP['model_preds'],preds)
    print "Successfully generated and saved model predictions."
    return preds

#########################
def add_unique_craters(tuple, crater_dist, thresh_longlat2, thresh_rad2):
    Long, Lat, Rad = crater_dist.T
    for j in range(len(tuple)):
        lo,la,r = tuple[j].T
        diff_longlat = (Long - lo)**2 + (Lat - la)**2
        Rad_ = Rad[diff_longlat < thresh_longlat2]
        if len(Rad_) > 0:
            diff_rad = ((Rad_ - r)/r)**2                #fractional radius change
            index = diff_rad < thresh_rad2
            if len(np.where(index==True)[0]) == 0:      #unique value determined from long/lat, then rad
                crater_dist = np.vstack((crater_dist,tuple[j]))
        else:                                           #unique value determined from long/lat alone
            crater_dist = np.vstack((crater_dist,tuple[j]))
    return crater_dist

#########################
def extract_crater_dist(CP, pred_crater_dist):
    
    id = np.load(CP['ids'])
    P = cPickle.load(open(CP['image_prop'], 'r'))
    
    #load/generate model preds
    try:
        preds = np.load(CP['model_preds'])
        print "Loaded model predictions successfully"
    except:
        print "Couldnt load model predictions, generating"
        preds = get_model_preds(CP)
    
    master_img_height_pix = 23040.  #number of pixels for height
    master_img_height_lat = 180.    #degrees used for latitude
    r_moon = 1737.4                 #radius of the moon (km)
    dim = float(CP['dim'])          #image dimension (pixels, assume dim=height=width), needs to be float
    thresh_longlat2, thresh_rad2 = CP['llt2'], CP['rt2']
    
    N_matches_tot = 0
    for i in range(CP['nimgs']):
        #print i, len(pred_crater_dist)
        coords = template_match_target(preds[i])
        if len(coords) > 0:
            P_ = P[id[i]]
            img_pix_height = float(P_['box'][2] - P_['box'][0])
            pix_to_km = (master_img_height_lat/master_img_height_pix)*(np.pi/180.0)*(img_pix_height/dim)*r_moon
            long_pix,lat_pix,radii_pix = coords.T
            radii_km = radii_pix*pix_to_km
            long_deg = P_['llbd'][0] + (P_['llbd'][1]-P_['llbd'][0])*(long_pix/dim)
            lat_deg = P_['llbd'][3] - (P_['llbd'][3]-P_['llbd'][2])*(lat_pix/dim)
            tuple_ = np.column_stack((long_deg,lat_deg,radii_km))
            N_matches_tot += len(coords)
            
            #only add unique (non-duplicate) values to the master pred_crater_dist
            if len(pred_crater_dist) > 0:
                pred_crater_dist = add_unique_craters(tuple_, pred_crater_dist, thresh_longlat2, thresh_rad2)
            else:
                pred_crater_dist = np.concatenate((pred_crater_dist,tuple_))

    np.save('%s/crater_dist_n%d.npy'%(CP['dir'],CP['nimgs']),pred_crater_dist)
    return pred_crater_dist

#########################
if __name__ == '__main__':
    # Arguments
    CP = {}
    CP['dir'] = 'datasets/rings/Test_rings'                     #exclude final '/' in path
    CP['datatype'] = 'test'
    CP['nimgs'] = 10016
    CP['model_preds'] = '%s/%s_modelpreds_n%d_final.npy'%(CP['dir'],CP['datatype'],CP['nimgs'])
    CP['ids'] = '%s/%s_id.npy'%(CP['dir'],CP['datatype'])       #contains lola_id for each corresponding image
    CP['image_prop'] = '%s/lolaout_%s.p'%(dir,datatype)         #contains long/lat/zoom properties of each image
    
    #Needed to generate model_preds if they don't exist yet
    CP['data_dir'] = '%s/%s_data.npy'%(CP['dir'],CP['datatype'])
    CP['model'] = 'models/unet_s256_rings_n112_L1.0e-05_D0.15.h5'
    CP['dim'] = 256
    CP['rescale'] = 1
    
    # Tuned Hyperparameters - Shouldn't really change
    CP['llt2'] = 0.6    #D_{L,L} from Silburt et. al (2017)
    CP['rt2'] = 0.6     #D_{R} from Silburt et. al (2017)

    pred_crater_dist = np.empty([0,3])
    pred_crater_dist = extract_crater_dist(CP, pred_crater_dist)
