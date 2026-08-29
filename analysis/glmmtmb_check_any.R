# glmmTMB refit for any layer or the pooled model (paper spec).
# Detects the pooled case from a multi-level reactive_peds column and
# then adds the layer fixed effect and the three interaction intercepts.
# Usage: Rscript analysis/glmmtmb_check_any.R <stats_combo dir>
suppressMessages(library(glmmTMB))
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
d <- read.csv(file.path(dir, "model_frame_combo.csv"), check.names = FALSE)
meta <- read.csv(file.path(dir, "model_frame_meta.csv"), header = FALSE, row.names = 1)
reference <- meta["reference", 1]
covs <- strsplit(meta["covariates", 1], ",")[[1]]
d$algorithm <- relevel(factor(d$algorithm), ref = reference)
d$seed <- factor(d$seed); d$cell_map <- factor(d$cell_map)
d$map_task <- factor(d$map_task)
for (cv in covs) d[[paste0("scale_", cv)]] <- as.numeric(scale(d[[cv]]))
scaled <- paste0("scale_", covs)
pooled <- ("reactive_peds" %in% names(d)) && (length(unique(d$reactive_peds)) > 1)
fixed <- paste("algorithm +", paste(scaled, collapse = " + "))
re <- "(1|seed) + (1|cell_map) + (1|map_task)"
if (pooled) {
  d$reactive_peds <- factor(d$reactive_peds)
  d$layer_seed <- factor(paste(d$reactive_peds, d$seed, sep = ":"))
  d$layer_map <- factor(paste(d$reactive_peds, d$cell_map, sep = ":"))
  d$layer_map_task <- factor(paste(d$reactive_peds, d$map_task, sep = ":"))
  fixed <- paste(fixed, "+ reactive_peds")
  re <- paste(re, "+ (1|layer_seed) + (1|layer_map) + (1|layer_map_task)")
}
form <- as.formula(paste("success ~", fixed, "+", re))
cat("fitting", if (pooled) "pooled" else "per-layer", "spec, n =", nrow(d), "\n")
fit <- glmmTMB(form, data = d, family = binomial,
               control = glmmTMBControl(
                 optCtrl = list(iter.max = 10000, eval.max = 10000)))
fe <- summary(fit)$coefficients$cond
write.csv(data.frame(term = rownames(fe), estimate = fe[, "Estimate"],
                     se = fe[, "Std. Error"],
                     ci_lo = fe[, "Estimate"] - 1.96 * fe[, "Std. Error"],
                     ci_hi = fe[, "Estimate"] + 1.96 * fe[, "Std. Error"]),
          file.path(dir, "glmmtmb_fixed_effects_nestedtask.csv"), row.names = FALSE)
v <- VarCorr(fit)$cond
write.csv(data.frame(component = names(v),
                     sd = sapply(v, function(x) attr(x, "stddev"))),
          file.path(dir, "glmmtmb_variance_components_nestedtask.csv"),
          row.names = FALSE)
cat("done\n")
