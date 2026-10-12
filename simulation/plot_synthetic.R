# Plot computed synthetic results only. Never uses clinical figure data.
# Usage: Rscript simulation/plot_synthetic.R /absolute/external/run-directory
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1) stop('Supply exactly one external run directory.')
out <- normalizePath(args[1], mustWork=TRUE)
script <- sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1])
root <- normalizePath(file.path(dirname(script), '..'))
if (out == root || startsWith(out, paste0(root, '/'))) stop('Output must be outside repository.')
read <- function(name) read.csv(file.path(out, paste0(name, '.csv')))
banner <- 'SYNTHETIC ONLY | NOT CLINICAL RESULTS | NO TREATMENT EFFECTS'
base <- theme_classic(base_size=11) + theme(plot.title=element_text(face='bold'), legend.position='bottom')
theme_set(base)
palette <- c(fixed='#556B86', free='#008A91', spline='#C66B36')
save_plot <- function(plot,name,w=10,h=7) {
  plot <- plot + plot_annotation(title=banner,
      caption='Computed synthetic run. Inspect provenance and convergence diagnostics before interpretation.')
  ggsave(file.path(out,paste0(name,'_SYNTHETIC.pdf')),plot,width=w,height=h)
}
loading <- read('measurement_loadings')
loading <- loading[tolower(as.character(loading$fixed)) == 'false', ] # Unit anchor is fixed.
scores <- read('latent_scores')
a <- ggplot(loading,aes(mean,indicator)) + geom_vline(xintercept=0,linetype=2,color='grey60') +
  geom_segment(aes(x=lower,xend=upper,yend=indicator),linewidth=.7) + geom_point(size=2.5) +
  labs(x='Unstandardized loading: mean and 95% equal-tail CrI',y=NULL,title='A. Estimated non-anchor loadings')
b <- ggplot(scores,aes(posterior_mean)) + geom_histogram(aes(y=after_stat(density)),bins=25,fill='#9DB6CB',color='white') +
  geom_density() + labs(x='Latent instability (SD)',y='Density',title='B. Score distribution')
c <- ggplot(scores,aes(posterior_sd)) + geom_histogram(aes(y=after_stat(density)),bins=25,fill='#9DB6CB',color='white') +
  labs(x='Posterior SD',y='Density',title='C. Conditional uncertainty')
save_plot(a/(b+c),'Figure1')
assoc <- read('association_curve')
f2 <- ggplot(assoc,aes(score,mean)) + geom_ribbon(aes(ymin=lower,ymax=upper),fill='#9DB6CB',alpha=.35) +
  geom_line(linewidth=1) + labs(x='Latent instability (SD)',y='Synthetic fall probability',title='Logistic association; covariates at centered means')
save_plot(f2,'Figure2',8,5)
curves <- read('probability_curves')
contrasts <- read('shift_contrasts')
a <- ggplot(curves,aes(score,mean,color=model,fill=model)) + geom_ribbon(aes(ymin=lower,ymax=upper),alpha=.08,color=NA) +
  geom_line(linewidth=.8) + scale_color_manual(values=palette)+scale_fill_manual(values=palette) +
  labs(x='Latent instability (SD)',y='Probability',title='A. Alternative probability profiles')
b <- ggplot(curves,aes(score,derivative_mean,color=model,fill=model)) +
  geom_ribbon(aes(ymin=derivative_lower,ymax=derivative_upper),alpha=.08,color=NA)+geom_line(linewidth=.8)+
  geom_hline(yintercept=0,linetype=2)+scale_color_manual(values=palette)+scale_fill_manual(values=palette)+
  labs(x='Latent instability (SD)',y='dP/ds',title='B. Derivative, not uncertainty in tau')
c <- ggplot(subset(contrasts,abs(shift-.3)<1e-8),aes(score,mean,color=model)) + geom_line(linewidth=.8) +
  scale_color_manual(values=palette)+labs(x='Latent instability (SD)',y='Probability contrast (percentage points)',title='C. Hypothetical shift of 0.3 SD')
save_plot(a/(b+c),'Figure3')
s2 <- ggplot(contrasts,aes(score,mean,color=factor(shift)))+geom_line()+facet_wrap(~model)+
  labs(x='Latent instability (SD)',y='Model-implied contrast (percentage points)',color='Shift (SD)')
save_plot(s2,'Shift_comparison',11,5)
cat('Synthetic figures saved outside repository:',out,'\n')
