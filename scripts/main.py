from gks2p import *


# basepath = '/home/georgioskeliris/Desktop/gkel@NAS/MECP2TUN/'
# basepath = '/mnt/sdd/MECP2TUN'
basepath = '/mnt/NAS_UserStorage/georgioskeliris/MECP2TUN/'
fastbase = '/mnt/4TB_SSD/GKSMSLAB/MECP2TUN/'

stavroula_classifier='/mnt/12TB_HDD_6/ThinkmateB_HDD6_Data/stavroula-skarvelaki/Classifier_contrast.npy'
'''
ds = getDataPaths('coh2','w11','contrast')
ds = getOneExpPath('coh1','w11','M19','contrast')
ds = getOneSesPath('coh1', 'w11', 'M10', 'ses1', 'OO')
'''
       
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M10', ses='ses1', experiment='DR') 
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M10', ses='ses1', experiment='OR2') 
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M10', ses='ses1', experiment='OO') 
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M11', experiment='contrast') 
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M11', experiment='OR') 
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M11', experiment='DR') 
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M11', experiment='OO') 
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M12', experiment='contrast')
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M18', experiment='contrast')
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M19', experiment='contrast')      
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M20', experiment='contrast')
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M22', experiment='contrast')  
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M25', experiment='contrast')  
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M91', experiment='contrast')  

ds = datasetQuery(cohort='coh1', week='w22', mouseID='M11', experiment='contrast')  
ds = datasetQuery(cohort='coh1', week='w22', mouseID='M12', experiment='contrast')  
ds = datasetQuery(cohort='coh1', week='w22', mouseID='M18', experiment='contrast')  
ds = datasetQuery(cohort='coh1', week='w22', mouseID='M19', experiment='contrast')  
ds = datasetQuery(cohort='coh1', week='w22', mouseID='M20', experiment='contrast')  
ds = datasetQuery(cohort='coh1', week='w22', mouseID='M22', experiment='contrast')  
ds = datasetQuery(cohort='coh1', week='w22', mouseID='M117', experiment='contrast')  

ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M24')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M73')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M77', ses='ses3')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M78')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M81')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M145')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M148')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M149', ses='ses3')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M156')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M159')

ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M24')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M145')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M148',ses='ses1')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M148',ses='ses2')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M148',ses='ses3')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M149')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M156', ses='ses1')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M159')

ds = datasetQuery(cohort='coh3', week='w11', mouseID='M415')
ds = datasetQuery(cohort='coh3', week='w11', mouseID='M416')
ds = datasetQuery(cohort='coh3', week='w11', mouseID='M417')
ds = datasetQuery(cohort='coh3', week='w11', mouseID='M418')
ds = datasetQuery(cohort='coh3', week='w11', mouseID='M419', ses='ses1')

ds = datasetQuery(cohort='coh1', week='w11', experiment='OR') 
ds = ds[ds.mouseID!='M10']
ds = ds[ds.mouseID!='M11'] # remove 'M11' that has already been processesed
ds = ds[ds.mouseID!='M12']
ds = ds[ds.mouseID!='M18']
ds = ds[ds.mouseID!='M20']

ds = datasetQuery(cohort='coh1', week='w11', experiment='SF')
ds = ds[ds.mouseID!='M10']
ds = datasetQuery(cohort='coh1', week='w11', experiment='SF', mouseID="M10")

ds = datasetQuery(cohort='coh1', week='w11', experiment='TF')
ds = datasetQuery(cohort='coh1', week='w11', experiment='TF', mouseID="M20")
ds=ds[ds.datID==42]

ds = datasetQuery(cohort='coh1', week='w22', experiment='OR')

ds = datasetQuery(cohort='coh4', mouseID="M489", experiment='contrast')
ds=ds[ds.datID==416]
ds=ds[ds.datID==417]

ds = datasetQuery(cohort='coh4', mouseID="M490", experiment='contrast')
ds=ds[ds.datID==418]
ds=ds[ds.datID==419]

ds = datasetQuery(cohort='coh4', mouseID="M493", experiment='contrast')

ds = datasetQuery(cohort='coh4', mouseID="M488", experiment='contrast')
ds=ds[ds.datID==415]

ds = datasetQuery(cohort='coh3', week='w22', mouseID="M419", experiment='contrast')

# ANALYSIS

# In case you need to change some of the default OPS parameters put them here:
db = {
    'pipeline': 'orig',
    'tau': 1.5,
    'spatial_scale': 0
}

# GET THE DATASET(S)


# Potentially check if corrupted
for d in range(0,len(ds)):
    dsetOK = checkDatasets(ds.iloc[d].rawPath)

# And make the ops from the .tif header (if not done)
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)


# this can load the ops instead of remaking it
# ops is a list of dictionaries
ops2 = gks2p_loadOps(ds, basepath)
        
# CONVERT TO BINARY
#basepath = '/home/georgioskeliris/Desktop/gkel@NAS/MECP2TUN/'
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None) # potentially can give a list of planes to process
gks2p_combine(ds, basepath)

gks2p_classify(ds, basepath)
gks2p_classify(ds, basepath, classfile=stavroula_classifier)
gks2p_deconvolve(ds,basepath,0.7) # run deconvolution with a different tau if necessary





# correct
gks2p_correctOpsPerPlane(ds, basepath)


stats = np.reshape(
    np.array([
        stat[j][k]
        for j in range(len(stat))
        for k in classifier['keys']
    ]), (len(stat), -1))
