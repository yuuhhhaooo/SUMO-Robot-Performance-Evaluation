# Profile confidence intervals for the variance components of the
# glmmTMB refit (nested spec). Wald intervals are a poor approximation
# for SD parameters, whose likelihood is asymmetric and bounded at
# zero, so the variance-component cross-check uses profile intervals.
# Usage: Rscript analysis/glmmtmb_profile.R results_sfm/peds_sfm/stats_combo

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
d$map_task <- factor(d$map_task)
for (cv in covs) {
  d[[paste0("scale_", cv)]] <- as.numeric(scale(d[[cv]]))
}
scaled <- paste0("scale_", covs)

form <- as.formula(paste(
  "success ~ algorithm +", paste(scaled, collapse = " + "),
  "+ (1|seed) + (1|cell_map) + (1|map_task)"))

cat("fitting nested specification...\n")
fit <- glmmTMB(form, data = d, family = binomial)

cat("likelihood-based intervals for the variance components...\n")
# uniroot inverts the likelihood-ratio test per SD parameter; it is the
# glmmTMB-recommended method for random-effect SDs and, unlike full
# profiling, copes with the near-zero map SD at the boundary
ci <- confint(fit, parm = "theta_", method = "uniroot")
# theta parameters are log-SDs: report on the SD scale, named by
# component (a boundary SD near zero yields NA interval bounds)
nm <- c("theta_1|seed.1" = "seed", "theta_1|cell_map.1" = "map",
        "theta_1|map_task.1" = "task")
out <- data.frame(component = nm[rownames(ci)],
                  sd_est = round(exp(ci[, 3]), 3),
                  sd_lo = round(exp(ci[, 1]), 3),
                  sd_hi = round(exp(ci[, 2]), 3))
write.csv(out, file.path(dir, "glmmtmb_vc_profile.csv"),
          row.names = FALSE)
print(out)
cat("done\n")
