# resources used: 
# library.virginia.edu
# cran.r-project.org

# loading the dataset
data <- read.table("http://static.lib.virginia.edu/statlab/materials/data/ami_data.DAT")
names(data) <- c("TOT","AMI","GEN","AMT","PR","DIAP","QRS")

# useful functions
summary(data)
pairs(data)

# machine learning model 1
mlm1 <- lm(cbind(TOT, AMI) ~ GEN + AMT + PR + DIAP + QRS, data = data)
summary(mlm1)

head(fitted(mlm1))
head(resid(mlm1))
sigma(mlm1)
coef(mlm1)
vcov(mlm1)

library(car)
Anova(mlm1)