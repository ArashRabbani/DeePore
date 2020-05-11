import DeePore as dp


Path_compact='Data\DeePore_Compact_Data.h5'
# Path_complete='..\..\..\BigData\DeePore\DeePore_Dataset.h5'
# dp.create_compact_dataset(Path_complete,Path_compact)
DataName=Path_compact
List,MIN,MAX=dp.prep(DataName)
TrainList, EvalList, TestList = dp.splitdata(List)
model=dp.trainmodel(DataName,TrainList,EvalList,MIN,MAX,retrain=0)  
# dp.testmodel(model,DataName,TestList,MIN,MAX)



# D1=px.readh5slice('Data.h5','X',[1]) 
# dp.show_feature_maps(D1)
# D1=px.readh5slice('Data.h5','Y',[1]) 
# a=0
# import os
# try:
#     os.remove('Data.h5')
# except:
#     pass
# for FileName in glob.glob(path):
#     A=dp.readsampledata(FileName)
#     B=dp.ecl_distance(A)
#     dp.writeh5slice(np.uint8(B*255),'Data.h5','X',Shape=[128,128,3])
#     C=px.readh5slice('Data/DeePore_Compact_Data.h5','Y',[a]) 
#     dp.writeh5slice(np.reshape(C,(1,1515,1)),'Data.h5','Y',Shape=[1515,1])
#     a=a+1
    



# D1=px.readh5slice('Data.h5','X',[1])  
# D2=px.readh5slice('Data.h5','Y',[1])  

# D=px.readh5slice('Data/DeePore_Compact_Data.h5','Y',[1])    
# DataName='Data/DeePore_Compact_Data.h5'
# dp.check_get('https://www.linktodata',DataName)               
# List,MIN,MAX=dp.prep(DataName)
# TrainList, EvalList, TestList = dp.splitdata(List)
# model=dp.trainmodel(DataName,TrainList,EvalList,MIN,MAX,retrain=0)  
# # #  Now Testing the Model on the test samples
# # dp.testmodel(model,DataName,TestList,MIN,MAX)
# # # dp.predict.absperm('')
# # A=dp.readsampledata(FileName='Data/Sample_gray.jpg')
# # A=dp.readsampledata(FileName='Data/Sample.jpg')
# A=dp.readsampledata(FileName='Data/Sample_large.mat')
# A=dp.readsampledata(FileName=r"A00013.npz")

# B=dp.ecl_distance(A)
# dp.show_feature_maps(B)
A=dp.readsampledata(FileName="Data/A.mat")
B=dp.ecl_distance(A)
single_values=dp.predict(model,B,MIN,MAX,res=4.8)
# import p,px
# px.im(B[0,:,:,1])
# # px.im(A[0,:,:,1])