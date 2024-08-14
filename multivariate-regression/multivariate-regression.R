# work in progress

# resources used:
# library.virginia.edu

library(car)

# loading the dataset
data <- read.table("http://static.lib.virginia.edu/statlab/materials/data/ami_data.DAT")
names(data) <- c("TOT", "AMI", "GEN", "AMT", "PR", "DIAP", "QRS")

# useful functions
summary(data)
pairs(data)

# machine learning model 1
mlm1 <- lm(cbind(TOT, AMI) ~ GEN + AMT + PR + DIAP + QRS, data = data)
summary(mlm1)

Anova(mlm1)
Manova(mlm1)

# machine learning model 2
mlm2 <- update(mlm1, . ~ . - PR - DIAP - QRS)
summary(mlm2)

Anova(mlm2)
Manova(mlm2)

anova(mlm1, mlm2)

sample <- data.frame(GEN = 1, AMT = 1200)
p <- predict(mlm2, sample)
p