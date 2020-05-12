import DeePore as dp
# Quick start: 
# Feed your porous material image to see its properties predicted 
# 1. load the trained model
model=dp.loadmodel()
# 2. read and transform your data into initial feature maps
# in this example, the data is a 400 x 400 x 400 binary image in which 0 indicates void and 1 indicates solid space
A=dp.feedsampledata(FileName="Data/Sample_large.mat")
# 3. show feature maps (optional)
dp.show_feature_maps(A)
# 4. predict properties
all_preds=dp.predict(model,A,res=4.8) # res is the spatial resolution of image in micron/pixel
# 5. save results into a text file and also print it in console
dp.prettyresult(all_preds,'results.txt')
