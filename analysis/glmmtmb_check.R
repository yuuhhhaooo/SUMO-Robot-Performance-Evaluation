# glmmTMB refit of the sfm success model (supervisor request, item 5).
# Reads the model frame exported by glmmtmb_check.py and fits the same
# specification as the thesis: success ~ combo + standardized task
# covariates, crossed random intercepts for seed, map, task.
# Also fits the map-nested task variant (item 6) for comparison.
# Usage: Rscript analysis/glmmtmb_check.R results_sfm/peds_sfm/stats_combo

suppressMessages(library(glmmTMB))

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[1] else "results_sfm/peds_sfm/stats_combo"

d <- read.csv(file.path(dir, "model_frame_sfm_combo.csv"),
              check.names = FALSE)
meta <- read.csv(file.path(dir, "model_frame_meta.csv"),
                 header = FALSE, row.names = 1)
reference <- meta["reference", 1]
covs <- strsplit(meta["covariates", 1], ",")[[1]]

d$algorithm <- relevel(factor(d$algorithm), ref = reference)
d$seed <- factor(d$seed)
d$cell_map <- factor(d$cell_map)
d$task <- factor(d$task)
d$map_task <- factor(d$map_task)
for (cv in covs) {
  d[[paste0("scale_", cv)]] <- as.numeric(scale(d[[cv]]))
}
scaled <- paste0("scale_", covs)

form_crossed <- as.formula(paste(
  "success ~ algorithm +", paste(scaled, collapse = " + "),
  "+ (1|seed) + (1|cell_map) + (1|task)"))
form_nested <- as.formula(paste(
  "success ~ algorithm +", paste(scaled, collapse = " + "),
  "+ (1|seed) + (1|cell_map) + (1|map_task)"))

cat("fitting crossed (paper) specification...\n")
fit <- glmmTMB(form_crossed, data = d, family = binomial)
cat("fitting map-nested task specification...\n")
fit_nested <- glmmTMB(form_nested, data = d, family = binomial)

fe <- summary(fit)$coefficients$cond
write.csv(data.frame(term = rownames(fe),
                     estimate = fe[, "Estimate"],
                     se = fe[, "Std. Error"],
                     ci_lo = fe[, "Estimate"] - 1.96 * fe[, "Std. Error"],
                     ci_hi = fe[, "Estimate"] + 1.96 * fe[, "Std. Error"]),
          file.path(dir, "glmmtmb_fixed_effects.csv"), row.names = FALSE)

vc_sd <- function(f) {
  v <- VarCorr(f)$cond
  data.frame(component = names(v),
             sd = sapply(v, function(x) attr(x, "stddev")))
}
write.csv(vc_sd(fit), file.path(dir, "glmmtmb_variance_components.csv"),
          row.names = FALSE)

fe2 <- summary(fit_nested)$coefficients$cond
write.csv(data.frame(term = rownames(fe2),
                     estimate = fe2[, "Estimate"],
                     se = fe2[, "Std. Error"],
                     ci_lo = fe2[, "Estimate"] - 1.96 * fe2[, "Std. Error"],
                     ci_hi = fe2[, "Estimate"] + 1.96 * fe2[, "Std. Error"]),
          file.path(dir, "glmmtmb_fixed_effects_nestedtask.csv"),
          row.names = FALSE)
write.csv(vc_sd(fit_nested),
          file.path(dir, "glmmtmb_variance_components_nestedtask.csv"),
          row.names = FALSE)

cat("AIC crossed:", AIC(fit), " nested:", AIC(fit_nested), "\n")
cat("done\n")
