import DeePore as dp
# Comparing statistics of the training, validation and testing data: 
# 1. check or download the compact data
Data_compact='Data\DeePore_Compact_Data.h5'
dp.check_get('https://zenodo.org/record/4297035/files/DeePore_Compact_Data.h5?download=1',Data_compact) 
# 2. prepare the dataset by removing outliers and creating list of training, validation and test samples
List=dp.prep(Data_compact)
TrainList, EvalList, TestList = dp.splitdata(List)

# 3. read datasets 'Y' into arrays
Data_Eval=dp.readh5slice(Data_compact,'Y',EvalList)
Data_Train=dp.readh5slice(Data_compact,'Y',TrainList)
Data_Test=dp.readh5slice(Data_compact,'Y',TestList)

# exporting to MATLAB for extra postprocessing if you needed
# import scipy.io as sio
# sio.savemat('All_Data.mat',{'train':Data_Train,'eval':Data_Eval,'test':Data_Test})
 
# 4. plot histograms
import matplotlib.pyplot as plt
FN=5 # feature id number, you can select 0 to 14
h=plt.hist(Data_Eval[:,FN,0],50,histtype='step',density=True,label='validation')
h=plt.hist(Data_Train[:,FN,0],50,histtype='step',density=True,label='training')
h=plt.hist(Data_Test[:,FN,0],50,histtype='step',density=True,label='testing')
plt.ylabel('Frequency')
plt.xlabel('Property #' +str(FN))
plt.legend(loc='upper left')
