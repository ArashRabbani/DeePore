import DeePore as dp
# Part #1    A quick start 
# Feed your porous material image to see its properties predicted 
# 1. load the trained model
model=dp.loadmodel()
# 2. read and transform your data into initial feature maps
# in this example, the data is a 400 x 400 x 400 binary MATLAB mat file in which 0 indicates void and 1 indicates solid space
A=dp.feedsampledata(FileName="Data/Sample_large.mat")
# 3. show feature maps (optional)
dp.show_feature_maps(A)
# 4. predict properties
all_preds=dp.predict(model,A,res=4.8) # res is the spatial resolution of image in micron/pixel
# 5. save results into a text file and also print it in console
dp.prettyresult(all_preds,'results.txt')



# Part #2    Compatibility
# 1. you can try to load numpy 3-d arrays with the same manner
A=dp.feedsampledata(FileName="Data/Sample.npy")
# 2. also 2-d images with formats of jpg and png are welcome
# if you import a 2-d image, the code creates 3 arbitrary mid-slices by flipping the 2-d image
A=dp.feedsampledata(FileName="Data/Sample.jpg")
A=dp.feedsampledata(FileName="Data/Sample.png")
# 3. when your image is larger than 256 x 256 x 256, the code automatically consider sliding windows to cover the whole image and report back to you the averaged predictions
A=dp.feedsampledata(FileName="Data/Sample_large.mat")
# when the data is loaded and transformed to the initial feature maps using this function, you are good to go and find its properties as shown above.
