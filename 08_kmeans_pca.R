#############################################################
# Machine Learning for Social Sciences
# Felix Hagemeister
# 2022

# R Script: K-Means and PCA

# clear environment
# -------------------------------
rm(list = ls())

# set custom path using system user name
# -------------------------------
if (Sys.getenv("USERNAME") == "felix"){
    setwd("C:/Users/felix/Dropbox/HfP/Teaching/WiSe21/ML/")}
if (Sys.getenv("USERNAME") == "[YOUR USER NAME HERE]"){
    setwd("[YOUR PATH HERE")}

# load packages
# --------------------------------------------------


# load European Protein Consumption, in grams/person-day data
# --------------------------------------------------

food <- read.csv("data/protein.csv", row.names=1) # 1st column is country name

head(food)

plot(density(food$Milk), main="Density Milk consumption [daily grams p.c.]")


# Pre-process: scale the data
# --------------------------------------------------

?scale()
xfood <- scale(food) 

# (1) K-Means
# --------------------------------------------------

# for K=3
(grpMeat <- kmeans(xfood,  centers=3, nstart=10))

plot(xfood[,"RedMeat"], xfood[,"WhiteMeat"], xlim=c(-2,2.75), 
    type="n", xlab="Red Meat", ylab="White Meat", bty="n")
text(xfood[,"RedMeat"], xfood[,"WhiteMeat"], labels=rownames(food), 
    col=rainbow(3)[grpMeat$cluster])

# for K=7 (clustering on all protein groups)
grpProtein <- kmeans(xfood, centers=7, nstart=50)
grpProtein

par(mai=c(.9,.9,.1,.1))
plot(xfood[,"RedMeat"], xfood[,"WhiteMeat"], xlim=c(-2,2.75), 
    type="n", xlab="Red Meat", ylab="White Meat", bty="n")
text(xfood[,"RedMeat"], xfood[,"WhiteMeat"], labels=rownames(food), 
    col=rainbow(7)[grpProtein$cluster]) ## col is all that differs from first plot

plot(xfood[,"RedMeat"], xfood[,"Fish"], xlim=c(-2,2.75), 
    type="n", xlab="Red Meat", ylab="Fish")
text(xfood[,"RedMeat"], xfood[,"Fish"], labels=rownames(food), 
    col=rainbow(7)[grpProtein$cluster]) ## col is all that differs from first plot

# (2) Factor Models: Principal Component Analysis (PCA)
# -----------------------------------------------------------

?prcomp()
pcfood <- prcomp(food, scale=TRUE)

# rotation matrix (phi)
round(pcfood$rotation, 1)

# PC scores (nu)
round( predict(pcfood, newdata=food["France",]),2)
head( round(zfood <- predict(pcfood),1)) 

## predict is just doing the same thing as the below:
z <- scale(food)%*%pcfood$rotation
all(z==zfood)

## implies rotations are on scale of standard deviations if scale=TRUE
## looks like PC1 is an 'average diet', PC2 is iberian
t( round(pcfood$rotation[,1:2],2) )

## do some k-means, for comparison
grpProtein <- kmeans(scale(food), centers=7, nstart=20)

## how do the PCs look?
par(mfrow=c(1,2))
plot(zfood[,1:2], type="n", xlim=c(-4,5))
text(x=zfood[,1], y=zfood[,2], labels=rownames(food), col=rainbow(7)[grpProtein$cluster])
plot(zfood[,3:4], type="n", xlim=c(-3,3))
text(x=zfood[,3], y=zfood[,4], labels=rownames(food), col=rainbow(7)[grpProtein$cluster])

## how many do we need? tough to tell, but can graph proportion of variation ("Screeplot")
par(mfrow=c(1,1))
plot(pcfood, main="")
mtext(side=1, "European Protein Principle Components",  line=1, font=2)

## summary puts these scree plots on a more intuitive scale: 
	## proportion of variation explained.
summary(pcfood)

### END
