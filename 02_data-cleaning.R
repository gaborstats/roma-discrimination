####################
# adattisztitas    #
####################

# TARTALOM
# 1. Unicode helyrellitasa
# 2. bulletpointos valaszok dummyzasa es torlese
# 3. hyperlinkek dummizasa


library(stringr)
library(tidytext)
library(syuzhet)
library(dplyr)
library(qdap) # to remove urls
library(magrittr) # a pipe fv-hez

# 0. adatok importalasa -------------------------------------------------

getwd()
setwd("C:/Users/Gabor/Documents/01_ELTE/00_szakdoga/03_Adatok/05_Adatfeldolgozas/05_NLP-proba")

df_teljes = read.csv("2020-08-01_AB-text.csv", sep = ";", stringsAsFactors = F,
              header= T)

df = df_teljes[complete.cases(df_teljes$reply),]
rm(df_teljes)


# 1. Unicode karakterek helyrellitasa ------------------------------------

df$reply %<>%
  gsub("=C1", "á", .) %>%
  gsub("=C3=81", "á", .) %>%
  gsub("í<U+0096>", "ö", fixed = T, .) %>%
  gsub("Ĺą", "ű", .) %>%
  gsub("=fb", "ű", .) %>%
  gsub("=FB", "ű", .) %>%
  gsub("+D32", "", fixed = T, .) %>%
  gsub("=C5=B0", "ű", .) %>%
  gsub("=C5=90", "ő", .) %>%
  gsub("=E2=80=9E", "", .) %>%
  gsub("=C2=B7", "", .) %>%
  gsub("=e2=80=a6", "", .) %>%
  gsub("=C5=91", "ő", .) %>%
  gsub("=E2=80=9D", "", .)%>%
  gsub("=E2=80=A6", "", .)%>%
  gsub("=C3=93", "ó", .)%>%
  gsub("=96", "-", .)%>%
  gsub("=C3=89", "é", .)%>%
  gsub("=C5=B1", "ű", .)%>%
  gsub("=C2=A7", "", .)%>%
  gsub("=2E", "", .)%>%
  gsub("=C2=B0", "", .)%>%
  gsub("=E2=80=9C", "", .)%>%
  gsub("=A0", "", .)%>%
  gsub("=C3=BC", "ü", .) %>%
  gsub("Ĺ\u0090", "Ő", .) %>%
  gsub("í\u0081", "á", .) %>%
  gsub("Â", "", .) 

# NB: fixed = T, csak char-ra keres, nem hasznal regexet.
# grep("í<U+0096>", df$reply, fixed = T)


# 2. idezetek dummizasa ---------------------------------------------

# dummizzuk a valaszadokat, akik ideztek az emailembol es 
# bulletpoint-szeruen valaszoltak

a = grep("Van kifejezetten lakókocsik számára kijelölt parkolóhely", df$reply, value = F)
b = grep("hol található és milyen felszereltségű", df$reply, value = F)
c = grep("Legfeljebb hány éjszakát tölthetnek itt a látogatók", df$reply, value = F)
d = grep("Amennyiben a településen nincs kifejezetten", df$reply, value = F)
e = grep("tudna ajánlani egy vagy két látványosságot a környéken", df$reply, value = F)

# melyik valaszban van idezett szoveg? 
lst = c(a, b, c, d, e)
lst = unique(lst)

# dummizzuk ezeket a valaszokat
df$bulletP = 0
for (i in lst) {
  df[i,"bulletP"] = 1
}

rm(lst,a,b,c,d,e)

# 2.1. Whitespace kezelese

# Q: Most mert nem latni a line breakek-et a szovegben? 
#df[111:123, "reply"]
#df$reply[112]

# remove line breaks
df[, "reply"] = gsub("\r?\n|\r", " ", df[, "reply"])
# NB: \t stands for tab.

# remove multiple spaces
library(stringr)
df[, "reply"] = str_squish(df[, "reply"])



# 2.2. Sajat szoveg eltuntetese

grep("Van kifejezetten lakókocsik számára kijelölt parkolóhely az önök településén?", df$reply, value = F)
# hurra!
# NB: kell a \\ escape character a ? ele, ha azt is ki akarjuk venni. 
df$reply = gsub("Van kifejezetten lakókocsik számára kijelölt parkolóhely az önök településén\\?", "", df$reply,ignore.case = T)
df$reply = gsub("Van kifejezetten lakókocsik számára kijelölt parkolóhely darnón\\?", "", df$reply, ignore.case = T)
grep("Van kifejezetten lakókocsik ", df$reply, value = F)


#df[121, "reply"] # Darno
#df[4, "reply"] 


#grep("Ha igen, hol található és milyen felszereltségű\\?", df$reply, value = F)
df$reply = gsub("Ha igen, hol található és milyen felszereltségű\\?", "", df$reply, ignore.case = T)

#grep("Legfeljebb hány éjszakát tölthetnek itt a látogatók\\?", df$reply, value = F)
df$reply = gsub("Legfeljebb hány éjszakát tölthetnek itt a látogatók\\?", "", df$reply, ignore.case = T)

#grep("Amennyiben a településen nincs kifejezetten lakókocsik számára kijelölt parkolóhely, tudna tanácsod adni, hol érdemes parkolnunk a lehető legközelebb a településközponthoz\\?", df$reply, value = F)
df$reply = gsub("Amennyiben a településen nincs kifejezetten lakókocsik számára kijelölt parkolóhely, tudna tanácsod adni, hol érdemes parkolnunk a lehető legközelebb a településközponthoz\\?", "", df$reply, ignore.case = T)
df$reply = gsub("nincs kifejezetten lakókocsik számára kijelölt parkolóhely, tudna tanácsod adni, hol érdemes parkolnunk a lehető legközelebb a településközponthoz\\?", "", df$reply, ignore.case = T)

#grep("Végezetül, tudna ajánlani egy vagy két látványosságot a környéken, amit semmiképpen sem érdemes kihagynunk, vagy egy olyan helyet, ahol részletes információt kaphatunk a település történetéről\\?", df$reply, value = F)
df$reply = gsub("Végezetül, tudna ajánlani egy vagy két látványosságot a környéken, amit semmiképpen sem érdemes kihagynunk, vagy egy olyan helyet, ahol részletes információt kaphatunk a település történetéről\\?", "", df$reply, ignore.case = T)

# 2.3. Whitesapce hozzadasa, ahova kell 

# szokoz beillesztese kozpontozas utan
# ha nincs, akkor itt szokozt illesztunk be a pont es betu/szam koze
df$reply <- gsub("\\.(?=[a-zA-Zá-űÁ-Ű0-9])", replacement = "\\. ", 
                                   x=df$reply, perl = T)
# Q2: Ahhol ez gondot okoz/hibasan valaszt szet mondatot:
# utcaneveknel (Pl. Kolcsey u.14 Balatonfenyvesen)
# honlapoknal, ha lehagytak a "www."prefixet es ezert nem sikerult kiszurni (Pl. adand.hu)
# titulusoknal (Pl. dr.Szalai Miklos emlekmu, Halimbán.)
# ezresek tagolasanal (.000 Ft)

# itt pedig szokoz a vesszo es betu koze
df$reply <- gsub("\\,(?=[a-zA-Zá-űÁ-Ű0-9])", replacement = "\\, ", 
                 x=df$reply, perl = T)

# perjel utan is tegyunk be egy szokozt
df$reply <- gsub("/(?=[a-zA-Zá-űÁ-Ű0-9])", replacement = "/ ", 
                 x=df$reply, perl = T)

# zarojel elott es utan is tegyunk be egy szokozt
#df[27,"reply"]
df$reply <- gsub("(?<=[a-zA-Zá-űÁ-Ű0-9])\\(", replacement = " (", 
                 x=df$reply, perl = T)
df$reply <- gsub("\\)(?=[a-zA-Zá-űÁ-Ű0-9])", replacement = ") ", 
                x=df$reply, perl = T)

# 3. hyperlinkek dummyzasa ----------------------------------------------

library(qdap) # to remove urls
a = grep( ("www|http"), df$reply, value = F)
b = grep("HYPERLINK", df$reply, value = F)

# dummizzuk a hiperlinkeket (NB: lehetne a frequencit is szamolni!)
lst = c(a, b)
lst = unique(lst)

df$hyperlink = 0
for (i in lst){
  df[i,"hyperlink"] = 1
}

rm(lst,a,b)

# hiperlink- es emailstring es csonkok eltuntetese az arulkodo karakterek alapjan, 
# ez esetben mukodik, de nem univerzalis megoldas.  
# grep("<", df$reply, value = F)
# grep("=", df$reply, value = F)

df$reply <- gsub("(\\s)((\\S+)?=.*?)(?=\\s)", replacement = "", 
                 x=df$reply, perl = T)
df$reply <- gsub("(\\s)((\\S+)?<.*?)(?=\\s)", replacement = "", 
                 x=df$reply, perl = T)
df$reply <- gsub("(\\s)((\\S+)?>.*?)(?=\\s)", replacement = "", 
                 x=df$reply, perl = T)


# vegyuk ki a HYPERLINK stringet
df$reply <- gsub("HYPERLINK", replacement = "", 
                 x=df$reply, perl = T)




setwd("C:/Users/Gabor/Documents/01_ELTE/00_szakdoga/03_Adatok/05_Adatfeldolgozas/05_NLP-proba")
write.table(df, '2020-08-30_reply.csv', 
            sep = ";", dec = ",", na = "NA", quote = T, qmethod = c("double"), 
            row.names = F, fileEncoding = "latin2")


