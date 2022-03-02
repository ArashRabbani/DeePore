import DeePore as dp
# Explore the dataset: 
# If you want to open images of dataset and visualize them
# 1. check or download the complete dataset 
Data_complete='Data\DeePore_Dataset.h5'
dp.check_get('https://zenodo.org/record/4297035/files/DeePore_Dataset.h5?download=1',Data_complete)
# 2. read the first image out of 17700
A=dp.readh5slice(Data_complete,'X',[0]) 
# 3. show mid-slices of the loaded image
dp.showentry(A)
# 4. show and save the properties of this image which assumed to be the ground truth as text file
props=dp.readh5slice(Data_complete,'Y',[0])
dp.prettyresult(props,'sample_gt.txt',units='px')


