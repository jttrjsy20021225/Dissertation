library(DogsTrustCBR)
library(dplyr)  
library(tibble) 

# This code does not classify all dogs. Some categories were classified manually by Lauren.

df <- read.csv("dog_trust_data_updated.csv", stringsAsFactors = FALSE)
kc_meta <- read.csv("kc_metadata.csv", stringsAsFactors = FALSE)


data(package = "DogsTrustCBR")


# See the avaiable list
available_lists <- get_breed_list()
print(available_lists)


# Clean the dataset
df <- df %>%
  clean_breeds(
    varname        = "breed",
    write_unmatched= FALSE,
    silent         = TRUE,
    unmatched      = "other",
    keep_original  = TRUE
  )


# Combine the kc_group to left_joint
df2 <- df %>%
  left_join(
    kc_meta,
    by = c("breed_cleaned" = "breed_matching")
  )


# Transfer kc_group to factor
df2 <- df2 %>%
  mutate(
    kc_group = factor(kc_group)
  )


write.csv(df2, "dog_trust_data_updated_breed.csv", row.names = FALSE)






