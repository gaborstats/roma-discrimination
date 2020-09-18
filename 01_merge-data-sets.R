###############################
# adatbazisok osszkapcsolasa  #
###############################

# TARTALOM
# 1. az adatbazist kiegeszitettem az ocr utjabn pdf-bol kinyert valaszokkal
# 2. elmentettem a valaszokat nehany egyeb valtozoval egyutt

getwd()
setwd("C:/Users/Gabor/Documents/01_ELTE/00_szakdoga/03_Adatok/06_Eredmenyek")
df_oreg =read.csv("2020-03-19_AB-adatbazis_jav.csv", sep = ";", stringsAsFactors = F)

library("readxl")
df_masik = read_excel("2020-08-01_AB-adatbazis_ocr-updated.xlsx")


#install.packages("tidyverse")
library("tidyverse")
df_teljes = full_join(x = df_oreg, y = df_masik[,4:5], by = "telepules")
rm(df_oreg, df_masik)

# valaszuk ki a karakter valaszt es a featuroket
df = select(df_teljes, reply, roma_felado, magas_statusz, telepules, onk.hiv_szekhely)
rm(df_teljes)

setwd("C:/Users/Gabor/Documents/01_ELTE/00_szakdoga/03_Adatok/05_Adatfeldolgozas/05_NLP-proba")
write.table(df, '2020-08-01_AB-text.csv', sep = ";", dec = ",", na = "NA",
            quote = T, qmethod = c("double"), row.names = F)



