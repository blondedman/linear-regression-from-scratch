# resources used: 
# library.virginia.edu
# www.datacamp.com

# loading the dataset
data <- read.table("http://static.lib.virginia.edu/statlab/materials/data/ami_data.DAT")
names(data) <- c("TOT","AMI","GEN","AMT","PR","DIAP","QRS")

# useful functions
summary(data)
pairs(data)

# machine learning model 1
mlm1 <- lm(cbind(TOT, AMI) ~ GEN + AMT + PR + DIAP + QRS, data = data)
summary(mlm1)

# residual values
head(resid(mlm1))

# fitted values
head(fitted(mlm1))

# coefficient values
coef(mlm1)

# residual standard error 
sigma(mlm1)

# variance-covariance matrix
vcov(mlm1)

