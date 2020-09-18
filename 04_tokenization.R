# TARTALOM

# 1. mondatok kinyerese
# 2. tokenizalas

library(stringr)
library(tidytext)
library(syuzhet)
library(dplyr)
library(magrittr) # a pipe fv-hez
library(tibble)

# 0. adatok importalasa -------------------------------------------------

getwd()
setwd("C:/Users/Gabor/Documents/01_ELTE/00_szakdoga/03_Adatok/05_Adatfeldolgozas/05_NLP-proba")

df = read.csv("2020-09-02_lemmatized.csv", sep = ";", stringsAsFactors = F,
                     header= T)

df$reply = NULL


# 1. split to sentences --------------------------------------------------

# NB: itt fontos, h a mondatok nagy kezdobetuvel kezdodjenek

require(tidytext)
df = as_tibble(df)

Get_sentences <- df %>%
  unnest_sentences(output = sentence, input = lemma, strip_punct = T,
                   drop = F) %>% mutate(id = row_number())

Get_sentences$sent_len = sapply(Get_sentences$sentence, nchar)
Get_sentences <- Get_sentences[order(Get_sentences$sent_len, Get_sentences$sentence),]

# names(Get_sentences)


# "dr, udv, nincs" maradjon, a tobbi 10 karakteralatti mondatot pedig toroljuk.
maradjon = c("dr", "üdv", "nincs")

keep <- apply(Get_sentences["sentence"], 1, function(x) any(x %in% maradjon))

Get_sentences = Get_sentences[(Get_sentences$sent_len>=10| keep),]


# 1.2. telepulesnev es kozos onkori nevenek eltuntese -----------------------

telepulesnev_eltavolito <- function(n_nev = Get_sentences$telepules, 
                                    n_mondat = Get_sentences$sentence) {
  y = rep("", length(n_nev))
  for (i in 1:length(n_nev)) {
    y[i] = gsub( paste0(n_nev[i], "|", "((?<=", n_nev[i], ")\\S+)"), 
                 replacement = "", n_mondat[i], ignore.case = T,
                 perl = T)
  }
  return(y)
}


Get_sentences["sentence2"] = telepulesnev_eltavolito(
  n_nev = Get_sentences$telepules,
  n_mondat = Get_sentences$sentence)

Get_sentences["sentence3"] = telepulesnev_eltavolito(
  n_nev = Get_sentences$onk.hiv_szekhely, 
  n_mondat = Get_sentences$sentence2)

#Get_sentences$sentence2 = str_squish(Get_sentences$sentence)

Get_sentences <- subset(Get_sentences, select = -c(sentence, sentence2))
names(Get_sentences)[names(Get_sentences) == 'sentence3'] <- 'sentence'

# karakterhossz frissitese 
Get_sentences$sent_len = sapply(Get_sentences$sentence, nchar)


# 2. tokenizalas ---------------------------------------------------------

Get_sentences = as_tibble(Get_sentences)

Get_tokens <- Get_sentences %>%
  unnest_tokens(output = word, input = sentence, to_lower = T, drop = F)


# opcionalis: remove stop words
#my_stopwords <- tibble(word = c("a", "az", "és", "hogy"))

# egyeb stop szo jeloltek: 
# "ez", "is", "ahol", "alábbi", "egy", de, mely, meg, fel, vagy, ily, sem?

#Get_tokens = Get_tokens %>%
 # anti_join(my_stopwords, by = c("word" = "word"))

Get_tokens <- Get_tokens[order(Get_tokens$id),]


# 4. export results ----------------------------------------------------

# par percig tart:
write.table(Get_tokens, '2020-09-02_tokens.csv', 
            sep = ";", dec = ",", na = "NA", quote = T, qmethod = c("double"), 
            row.names = F, fileEncoding = "latin2")


Get_tokens_short <- subset(Get_tokens, select = -c(lemma, sent_len,
                                                   onk.hiv_szekhely))
# nagyobb adatoknal pickle-zni fog kelleni
write.table(Get_tokens_short, '2020-09-02_tokens_short.csv', 
            sep = ";", dec = ",", na = "NA", quote = T, qmethod = c("double"), 
            row.names = F, fileEncoding = "latin2")

# mehet a modellezes.
