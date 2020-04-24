# https://cran.r-project.org/web/packages/text2vec/vignettes/text-vectorization.html
# https://www.kaggle.com/therohk/million-headlines
# https://www.kaggle.com/therohk/india-headlines-news-dataset
library(text2vec)
library(data.table)
library(magrittr)
library(glmnet)
library(dplyr)

# raw.meals <- read.csv("/Users/diagdavenport/Desktop/LMS test/Last meal data - Sheet1.csv")
# raw.meals$year.split <- raw.meals$Year>=median(raw.meals$Year)
# mean(raw.meals$year.split)
# raw.meals$Requested.Meal <- as.character(raw.meals$Requested.Meal)
#
# raw.news <- read.csv('/Users/diagdavenport/Desktop/LMS test/abcnews-date-text.csv')
#
# raw.news <- raw.news[!is.na(raw.news$publish_date),]
#
# ______________________
#
# mob.data <- read.csv("/Users/diagdavenport/Desktop/LMS test/Talhelm Mobility.csv")
# mob.data$response <- as.character(mob.data$response)
#
#
#
#
# sample.ids <- sample(1:nrow(raw.news), 50000)
#
# raw.news <- raw.news[sample.ids,]
# raw.news$headline_text <- as.character(raw.news$headline_text)
# raw.news$clean.date <- as.Date(as.character(raw.news$publish_date), format = "%Y%m%d")
# raw.news$wkday.num <- as.POSIXlt(raw.news$clean.date)$wday
# raw.news$year <- as.numeric(format(raw.news$clean.date,'%Y'))
#
# raw.news$bad.year <- raw.news$year %in% c(2008,2009, 2010, 2011)
# raw.news$weekend <- raw.news$wkday.num %in% c(0,6)
#
# #~~~~~~~~~~~~~~~~~~

#' Run the lms test on a textual outcome
#'
#' @param text A character vector.
#' @param condition A binary vector.
#' @return A list of stuff detailing how predictive \code{text} is for \code{condition}.
#' @export
lms.test <- function(text, condition, data, iter = 1000, term.count.min = 10) {
  val <- is.character(data[[text]])
  #*make sure that text is character*#
  #val2 <- #*make sure that condition is binary or restricted otherwise*#
  #* look at iter and nrow and decide to throw a warning or not
  #*check that all the vars are good early*#kk
  ## first set up the playground
  setDT(data)
  # set.seed(2017L) #*how should i think about replicability?*#
  all_ids = 1:nrow(data)

  #*generate a warning if n is below some number?*#
  #* shoul i also allow for removal of obvious words?*#

  # separate test from train. require atleast 40 obs in test set, so CLT can kick in
  test.set.size = max(15, round(nrow(data)*0.2))
  test_ids = sample(all_ids, test.set.size)
  train_ids = setdiff(all_ids, test_ids)
  train = data[train_ids,]
  test = data[test_ids,]

  # set up some helper functions for the nlp pipeline
  prep_fun = tolower
  tok_fun = word_tokenizer

  it_train = itoken(train[[text]],
                    preprocessor = prep_fun,
                    tokenizer = tok_fun,
                    ids = train$id,
                    progressbar = FALSE)

  it_test = tok_fun(prep_fun(test[[text]]))
  it_test = itoken(it_test, ids = test$id, progressbar = FALSE)

  # Now set up the DTM (sparsely!), 1 and 2 grams for now
  cat("Generating document text matrix.\n")
  vocab = create_vocabulary(it_train, ngram = c(1L, 2L))

  vocab = prune_vocabulary(vocab, doc_count_min = term.count.min,
                           doc_proportion_max = 0.8)
  #*should doc_proportion_max be set more intelligently?*#
  #* should i be removing stop words?*#

  bigram_vectorizer = vocab_vectorizer(vocab)
  dtm_train = create_dtm(it_train, bigram_vectorizer)

  # Now reweight the DTM to boost signal
  cat("Reweighting document text matrix with TF-IDF algorithm.\n")
  tfidf = TfIdf$new()
  dtm_train_tfidf = fit_transform(dtm_train, tfidf)

  dtm_test_tfidf = create_dtm(it_test, bigram_vectorizer)
  dtm_test_tfidf = transform(dtm_test_tfidf, tfidf)

  # Now build a true predictor
  cat("Building classifier.\n")
  glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, y = train[[condition]],
                                family = 'binomial',
                                alpha = 1,
                                nfolds = 5,
                                thresh = 1e-3,
                                maxit = 1e3)

  calc_cross_ent_loss <- function(predictions, actuals) {
    loss_vec <- -(actuals*log(predictions) + (1- actuals)*log(1- predictions))
    mean_loss <- mean(loss_vec)
    std_loss <- sqrt(var(loss_vec))/sqrt(length(loss_vec))
    return(list(mean=mean_loss, std.err=std_loss))
  }

  ## Then store some outputs of the model
  #plot(glmnet_classifier)
  #print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
  real.preds = predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[,1]

  true.loss.vec <- calc_cross_ent_loss(predictions = real.preds, actuals = test[[condition]])
  true.loss.mean <- true.loss.vec$mean
  true.loss.std.err  <- true.loss.vec$std.err

  CF <- as.matrix(coef(glmnet_classifier, glmnet_classifier$lambda.1se))
  key.predictors <- CF[CF!=0,]

  # Finally, in order to do inference, we need to generate an empirical null distribution
  loss.holder <- c(NULL)

  #* Generalize to more than two classes*#
  #*spit out standard errors for the true loss?*#
  #*print the progress every 25 iters? Allow this to be set by user*#
  cat("Building empirical null distribution.\n")

  cat("Starting iteration 1. Will update in increments of 25...\n")
  for (i in 1:iter){
    if (mod(i,25) == 0) {
      cat("Checkpoint: iteration", i, "of", iter, "\n")
    }
    glmnet_classifier.iter <- cv.glmnet(x = dtm_train_tfidf, y = sample(train[[condition]]),
                                        family = 'binomial',
                                        alpha = 1,
                                        nfolds = 5,
                                        thresh = 1e-3,
                                        maxit = 1e3)

    null.preds = predict(glmnet_classifier.iter, dtm_test_tfidf, type = 'response')[,1]

    temp.loss.vec <- calc_cross_ent_loss(predictions = null.preds, actuals = test[[condition]])
    #temp.loss <- mean((null.preds-test[[condition]])^2)
    loss.holder <- c(loss.holder, temp.loss.vec$mean)

  }
  cat("Done!\n")

  p.val <- ecdf(loss.holder)(true.loss.mean)

  cat("Null hypothesis: Equal distribution of words between the two conditions.\n")
  cat("P-value: ", p.val, "\n")
  cat("The estimated loss is ", true.loss.mean)

  #*sort coefficients by p value*#
  sorted.features <- names(sort(abs(key.predictors[key.predictors != 'Intercept'])))
  sorted.features <- gsub("_", " ", sorted.features)

  key.predictor.frame <- NULL # just in case there aren't any

  if (length(sorted.features)>0) {
    out <- sapply(sorted.features, function(x) data %>% mutate(flag=row_number()%in%grep(x, data[[text]]))
                  %>% group_by(data[[condition]]) %>% summarise(m=mean(flag)))

    #*rename vars here*#
    key.predictor.frame <- as.data.frame(t(as.data.frame(out[2,])))
    key.predictor.frame$ratio <-  key.predictor.frame$V2/ key.predictor.frame$V1
  }

  return(list(p.val=p.val, true.loss.mean=true.loss.mean, true.loss.std.err=true.loss.std.err
              , key.predictors=key.predictor.frame, emp.null=loss.holder, dtm=dtm_train_tfidf))
}

# t1 = Sys.time()
# lms.result <- lms.test(text = 'Requested.Meal', condition = 'year.split', data = raw.meals, iter = 150)
# print(difftime(Sys.time(), t1, units = 'sec'))
#
# t1 = Sys.time()
# lms.result.weekend <- lms.test(text = 'headline_text', condition = 'weekend', data = raw.news, iter = 150)
# lms.result.badyear <- lms.test(text = 'headline_text', condition = 'bad.year', data = raw.news, iter = 150)
# print(difftime(Sys.time(), t1, units = 'sec'))
#
# lms.result.mob <- lms.test(text = 'response', condition = 'stable', data = mob.data, iter = 150, term.count.min = 2)
