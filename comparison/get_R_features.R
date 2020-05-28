library(data.table)
library(tsfeatures)

set.seed(12398)

dt <- fread("train.csv")
ids <- dt[, unique(unique_id)]
sample_ids <- sample(ids, 100)
setkey(dt, unique_id)
sample_series <- dt[.(sample_ids)]
fwrite(sample_series, "sample_series.csv")

series_list <- split(sample_series, by = "unique_id", keep.by = FALSE)
series_list <- lapply(series_list,
                      function(serie) serie[, ts(y, frequency = 7)])

features <- fread("funcs.txt")[, func]

series_features <- tsfeatures(series_list, features = features)
setDT(series_features)
series_features[, unique_id := names(series_list)]
fwrite(series_features, "Rfeatures.csv")
