import DeePore as dp
DataName='Data/DeePore_Compact_Data.h5'
dp.check_get('https://www.linktodata',DataName)               
List,MIN,MAX=dp.prep(DataName)
TrainList, EvalList, TestList = dp.splitdata(List)
model=dp.trainmodel(DataName,TrainList,EvalList,MIN,MAX,retrain=0)  
# #  Now Testing the Model on the test samples
# dp.testmodel(model,DataName,TestList,MIN,MAX)

# # dp.predict.absperm('')
# A=dp.readsampledata(FileName='Data/Sample_gray.jpg')
# A=dp.readsampledata(FileName='Data/Sample.jpg')
 

A=dp.readsampledata(FileName='Data/Sample_large.mat')
B=dp.ecl_distance(A)
dp.show_feature_maps(B)
single_values=dp.predict(model,B,MIN,MAX,res=4.8)

import p,px
px.im(B[0,:,:,1])
# px.im(A[0,:,:,1])