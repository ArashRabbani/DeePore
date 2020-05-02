options("repos" = c(CRAN = "http://cran.r-project.org/"))
if (!requireNamespace("hdf5r", quietly = TRUE))
  install.packages("hdf5r")

install.packages("ggfortify")


library("hdf5r")

setwd("C:/Users/rabar/Dropbox (The University of Manchester)/AR1/Porexa/Git/DeePore/Porous Material Characterization")
h <- H5File$new("Data2.h5", mode="r")
print(h)
Y <- h[["Y"]]


B=Y[1,1:15,]
B=B[,colSums(is.na(B))==0]
B=B[,colSums(is.infinite(B))==0]

B=t(as.matrix(B))
dim(B)



df <- data.frame(B) 
T=prcomp(df,scale = TRUE)
plot(T)

library(ggfortify)
autoplot(T)


P=T$x
K=kmeans(P, 7)
autoplot(K,data=P)
autoplot(T$rotation)