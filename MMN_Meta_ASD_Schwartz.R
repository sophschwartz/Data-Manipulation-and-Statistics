#################################
# Meta-Analysis Code used for 
# Schwartz, S., Shinn-Cunningham, B., & Tager-Flusberg, H. (2018). 
# Meta-analysis and systematic review of the literature characterizing 
# auditory mismatch negativity in individuals with autism. 
# Neuroscience & Biobehavioral Reviews, 87, 106-117.
# 
# Loads in CSV with compiled means, variances, and sample sizes from prior publications.
# Computes:
#     Hedge's g, the biased standardized mean difference estimation (SMD) for Cohen's D effect size
#     Pooled deviation
#     Cochran's Q statistic (Cochran, 1954) and Higgin's I^2 (Higgens et al., 2003) to compute between-experiment heterogeneity
#     Egger's weighted linear regression intercept test (Egger et al., 1997) to measure publication bias 
#################################

# Setup
rm(list=ls())
library(ggplot2)
library(meta)
library(forestplot)
library(BSDA)

# Load in Data
mmn_data <- read.csv("/Users/sophieschwartz/Desktop/MMNMeta_Full_Dataset.csv")

# Subsection of Data for Analyses
# Focus on amplitude, counterbalanced experiments, with comparison between nonspeech and speech stimuli
amplitude <- subset(mmn_data,included_amp=="Y")
amp_counterbalanced <- subset(amplitude,Counterbal=="Y")
amp_counterbalanced_nonspeech <- subset(amp_counterbalanced,SorNS=="NS")
amp_counterbalanced_speech <- subset(amp_counterbalanced,SorNS=="S")

# Main SMD Meta-analysis
mc_amp_full <- metacont(Ne, Me, Se, Nc, Mc, Sc, sm="SMD",
                data=amp_counterbalanced,
)
mc_amp_nonspeech <- metacont(Ne, Me, Se, Nc, Mc, Sc, sm="SMD",
                        data=amp_counterbalanced_nonspeech,
)
mc_amp_speech <- metacont(Ne, Me, Se, Nc, Mc, Sc, sm="SMD",
                        data=amp_counterbalanced_speech,
)

# Summary Plot of SMD
labelforplot<-amp_counterbalanced$Published_Label
AMP_CI_L=amp_counterbalanced$Amp_CI_L
AMP_CI_U=amp_counterbalanced$Amp_CI_U
Nfull=amp_counterbalanced$Nfull
setEPS()
postscript("Amplitude_SMD_Composite.eps")
myplot=ggplot(amp_counterbalanced, aes(x=Age_Mean, y=Amp_ES_bias, shape=SorNS, color=SorNS)) + 
  geom_errorbar(aes(ymin=Amp_CI_L, ymax=Amp_CI_U), linetype=3,width=0.8)+
  geom_point(aes(size=Nfull), alpha=0.7) + scale_size(range = c(5, 13)) +
  geom_hline(yintercept = 0.25, colour="red") + geom_hline(yintercept=0.01, colour="sky blue")
myplot + theme_bw() 

# Graph all meta-analysis data, organized by age
postscript("MMN_Amplitude_Age_Rev.eps",width=15,height=25)
forest(mc_amp_full, comb.fixed=FALSE, comb.random=TRUE, 
       leftcols=c("Published.Label","n.e","n.c"), rightcols=c("effect","ci"), xlim=c(-3,3), 
       lab.e="ASD",xlab="Counterbalanced Studies (Amplitude): Standardized ASD-CON", font=12)

#################################
# Publication Bias
# Symmetrical funnel plots suggest no evidence of publication bias based 
# on amplitude effect sizes from the counterbalanced sample. 
# Egger’s regression tests: Intercept: 0.57 [95% standard error confidence interval: 
# −1.55–2.68], t=0.53, p>0.05. 
#################################
bias <- metabias(mc_amp_full, plotit=TRUE)
funnel(mc_amp_full,ylim=c(0.8,0),xlim=c(-3,3))

#################################
# Heterogeneity of Sample
# Cochran's Q statistic (Cochran, 1954) and Higgin's I^2 (Higgens et al., 2003) to compute between-experiment heterogeneity
#################################
mc_speech<-cbind('Speech',mc_amp_nonspeech$Q,mc_amp_nonspeech$df.Q, 
                 mc_amp_nonspeech$tau, mc_amp_nonspeech$H, mc_amp_nonspeech$I2, 
                 mc_amp_nonspeech$lower.I2, mc_amp_nonspeech$upper.I2, 
                 mc_amp_nonspeech$TE.random, mc_amp_nonspeech$lower.random, 
                 mc_amp_nonspeech$upper.random, mc_amp_nonspeech$zval.random, 
                 mc_amp_nonspeech$pval.random)

mc_nonspeech<-cbind('Nonspeech', mc_amp_nonspeech$Q, mc_amp_nonspeech$df.Q, 
                    mc_amp_nonspeech$tau, mc_amp_nonspeech$H, mc_amp_nonspeech$I2, 
                    mc_amp_nonspeech$lower.I2, mc_amp_nonspeech$upper.I2, 
                    mc_amp_nonspeech$TE.random, mc_amp_nonspeech$lower.random, 
                    mc_amp_nonspeech$upper.random, mc_amp_nonspeech$zval.random, 
                    mc_amp_nonspeech$pval.random)

hetero_table<-data.frame(rbind(mc_speech, mc_nonspeech))
colnames(hetero_table)<-c("Subtype", "Q","df","Tau2","H","I2","Lower","Upper","SMD","lower","upper","z-value","p-value")
write.table(hetero_table, file="mmn_meta_hetero_table.txt", append=FALSE, sep="\t", row.names=FALSE, col.names=TRUE)

#################################
# Linear Model Statistics
# Mean age of the ASD group accounted for 25% of the variance in MMN amplitude effect 
# size across experiments (R2=0.25, F(1,22)=7.28, p=0.01) and age significantly 
# predicted effect size (Beta=−0.03, p=0.01). Visual inspection of effect size 
# organized by mean age of the ASD group revealed that the youngest cohorts of 
# ASD subjects had MMN amplitudes that were smaller than TD listeners, while adult 
# cohorts of ASD subjects had MMN amplitudes equal to or larger than those of their TD peers.

# A similar linear regression analysis on the influence of verbal IQ explained only 3% 
# of the effect size variance (R2=0.03, F(1,19)=0.60, p=0.45) and did not predict effect size 
# values (Beta=−0.004, p=0.45). 

# In addition, when verbal IQ was included as a covariate in a 
# linear model that measured the degree to which age predicted effect size, the model accounted for 
# 24.7% of the effect size variance but was not statistically significant (R2=0.247, F(2,18)=2.95, p=0.08). 
# In this model, mean age still significantly predicted effect size (Beta=−0.03, p=0.03). 
# These results suggest that effect size differences across age cannot be explained solely by 
# differences in verbal IQ.
#################################

summary(lm(Amp_ES_bias ~ Age_Mean, data = amp_counterbalanced))
summary(lm(Amp_ES_bias ~ VIQ_AUT_M, data = amp_counterbalanced))
summary(lm(Amp_ES_bias~ Age_Mean + VIQ_AUT_M, data = amp_counterbalanced))

# Next steps: Can machine learning be used to identify the same publications that we identified manually?
# e.g., Bannach-Brown, A., Przybyła, P., Thomas, J. et al. Machine learning algorithms for systematic review: 
# reducing workload in a preclinical review of animal studies and reducing human screening error. 
# Syst Rev 8, 23 (2019). https://doi.org/10.1186/s13643-019-0942-7

