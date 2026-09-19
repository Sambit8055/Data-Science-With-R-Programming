# Healthcare Cost Analysis using R
# Synthetic industry-style portfolio dataset

library(tidyverse)
library(scales)
library(caret)
library(randomForest)

healthcare <- read.csv("healthcare_cost_analysis_dataset.csv")
healthcare <- distinct(healthcare)

# Data quality
str(healthcare)
summary(healthcare)
colSums(is.na(healthcare))

# Feature engineering
healthcare <- healthcare %>%
  mutate(
    Age_Band = case_when(Age < 30 ~ "18-29", Age < 45 ~ "30-44",
                         Age < 60 ~ "45-59", TRUE ~ "60+"),
    Cost_Per_Day = Total_Cost_USD / Length_of_Stay_Days,
    High_Cost_Flag = if_else(
      Total_Cost_USD >= quantile(Total_Cost_USD, .75),
      "High Cost", "Standard Cost")
  )

# Executive KPIs
kpis <- healthcare %>% summarise(
  Patients=n(),
  Average_Cost=mean(Total_Cost_USD),
  Median_Cost=median(Total_Cost_USD),
  Average_LOS=mean(Length_of_Stay_Days),
  Readmission_Rate=mean(Readmission_30_Days=="Yes")*100,
  High_Cost_Rate=mean(High_Cost_Flag=="High Cost")*100
)
print(kpis)

# Department analysis
department_cost <- healthcare %>% group_by(Department) %>%
  summarise(Patients=n(),Total_Cost=sum(Total_Cost_USD),
            Average_Cost=mean(Total_Cost_USD),
            Average_LOS=mean(Length_of_Stay_Days)) %>%
  arrange(desc(Average_Cost))
print(department_cost)

ggplot(department_cost,aes(reorder(Department,Average_Cost),Average_Cost))+
  geom_col()+coord_flip()+scale_y_continuous(labels=dollar_format())+
  labs(title="Average Healthcare Cost by Department",x=NULL,y="Average Cost")+
  theme_minimal()

# Condition analysis
condition_cost <- healthcare %>% group_by(Primary_Condition) %>%
  summarise(Patients=n(),Average_Cost=mean(Total_Cost_USD),
            Median_Cost=median(Total_Cost_USD),
            Average_LOS=mean(Length_of_Stay_Days)) %>%
  arrange(desc(Average_Cost))
print(condition_cost)

# Insurance analysis
insurance_cost <- healthcare %>% group_by(Insurance_Type) %>%
  summarise(Patients=n(),Average_Cost=mean(Total_Cost_USD),
            Total_Cost=sum(Total_Cost_USD))
print(insurance_cost)

ggplot(healthcare,aes(Insurance_Type,Total_Cost_USD))+
  geom_boxplot()+scale_y_continuous(labels=dollar_format())+
  labs(title="Healthcare Cost Distribution by Insurance Type",
       x="Insurance Type",y="Total Cost")+theme_minimal()

# Length of stay vs cost
ggplot(healthcare,aes(Length_of_Stay_Days,Total_Cost_USD))+
  geom_point(alpha=.3)+geom_smooth(method="lm")+
  scale_y_continuous(labels=dollar_format())+
  labs(title="Length of Stay vs Healthcare Cost",
       x="Length of Stay (Days)",y="Total Cost")+theme_minimal()

# Readmission analysis
readmission_cost <- healthcare %>% group_by(Readmission_30_Days) %>%
  summarise(Patients=n(),Average_Cost=mean(Total_Cost_USD),
            Average_LOS=mean(Length_of_Stay_Days))
print(readmission_cost)

# Correlation
numeric_data <- healthcare %>% select(Age,Length_of_Stay_Days,
  Procedures_Count,Medication_Count,Followup_Visits,
  Total_Cost_USD,Claim_Amount_USD)
print(round(cor(numeric_data),2))

# Predictive modeling: high-cost classification
model_data <- healthcare %>% select(
  High_Cost_Flag,Age,Gender,Insurance_Type,Primary_Condition,
  Admission_Type,Department,Length_of_Stay_Days,
  Procedures_Count,Medication_Count,Followup_Visits)

set.seed(42)
idx <- createDataPartition(model_data$High_Cost_Flag,p=.80,list=FALSE)
train <- model_data[idx,]; test <- model_data[-idx,]

rf_model <- randomForest(High_Cost_Flag~.,data=train,ntree=300,importance=TRUE)
pred <- predict(rf_model,test)
print(confusionMatrix(pred,test$High_Cost_Flag,positive="High Cost"))
varImpPlot(rf_model,main="Variable Importance - High Cost Prediction")

# Export management summaries
write.csv(kpis,"executive_kpis.csv",row.names=FALSE)
write.csv(department_cost,"department_cost_summary.csv",row.names=FALSE)
write.csv(condition_cost,"condition_cost_summary.csv",row.names=FALSE)
write.csv(insurance_cost,"insurance_cost_summary.csv",row.names=FALSE)
