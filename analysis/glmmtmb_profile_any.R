# Profile (uniroot) intervals for the variance components of the glmmTMB
# refit, nested spec, for any per-layer model frame (model_frame_combo.csv).
# Usage: Rscript analysis/glmmtmb_profile_any.R <stats_combo dir> <out csv>
suppressMessages(library(glmmTMB))

args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
out <- args[2]

d <- read.csv(file.path(dir, "model_frame_combo.csv"), check.names = FALSE)
meta <- read.csv(file.path(dir, "model_frame_meta.csv"),
                 header = FALSE, row.names = 1)
reference <- meta["reference", 1]
covs <- strsplit(meta["covariates", 1], ",")[[1]]

d$algorithm <- relevel(factor(d$algorithm), ref = reference)
d$seed <- factor(d$seed)
d$cell_map <- factor(d$cell_map)
d$map_task <- factor(d$map_task)
for (cv in covs) {
  d[[paste0("scale_", cv)]] <- as.numeric(scale(d[[cv]]))
}
scaled <- paste0("scale_", covs)

form <- as.formula(paste(
  "success ~ algorithm +", paste(scaled, collapse = " + "),
  "+ (1|seed) + (1|cell_map) + (1|map_task)"))

cat("fitting nested specification for", dir, "...\n")
fit <- glmmTMB(form, data = d, family = binomial)

sds <- sqrt(unlist(lapply(VarCorr(fit)$cond, function(v) v[1, 1])))
cat("point SDs:\n"); print(round(sds, 4))

cat("uniroot intervals...\n")
ci <- try(confint(fit, parm = "theta_", method = "uniroot"), silent = TRUE)
res <- data.frame(component = names(sds), sd_est = round(sds, 3),
                  sd_lo = NA_real_, sd_hi = NA_real_)
if (inherits(ci, "try-error")) {
  cat("uniroot failed:", attr(ci, "condition")$message, "\n")
} else {
  print(ci)
  for (k in seq_len(nrow(res))) {
    hit <- grep(paste0("\\|", res$component[k], "\\b"), rownames(ci))
    if (length(hit) == 1) {
      res$sd_lo[k] <- round(exp(ci[hit, 1]), 3)
      res$sd_hi[k] <- round(exp(ci[hit, 2]), 3)
    }
  }
}
write.csv(res, out, row.names = FALSE)
cat("wrote", out, "\n")
