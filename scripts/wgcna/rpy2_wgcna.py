import rpy2.robjects as robjects
robjects.r('install.packages("BiocManager", '
           'repos="http://cran.r-project.org")')       
robjects.r('BiocManager::install("WGCNA")')
robjects.r('library(WGCNA)')    