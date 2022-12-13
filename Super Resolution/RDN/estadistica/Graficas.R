
dataPSNR2 <- read.table(file = "C:\\Users\\Usuario iTC\\Music\\Utpl\\Bien2\\EDSR_Tensorflow-master\\estadisticas\\Totalpsnr.txt", sep = ',')
head(dataPSNR2)
plot(x = dataPSNR2$V1, y = dataPSNR2$V3,xlab = "Observaciones",ylab = "PSNR", main = "Dispercion de los Datos de la Metrica PSNR",col =4 ,pch = 19)

dataSMMI3 <- read.table(file = "C:\\Users\\Usuario iTC\\Music\\Utpl\\Bien2\\EDSR_Tensorflow-master\\estadisticas\\smmi.txt", sep = ',')
head(dataSMMI3)
plot(x = dataSMMI3$V1, y = dataSMMI3$V3,xlab = "Observaciones",ylab = "SSIM", main = "Dispercion de los Datos de la Metrica SSIM",col ="RED" ,pch = 19)
install.packages("dplyr")
library(dplyr)
arithmetic.mean <- function(x) {sum(x)/length(x)}
datos<-c(4,7,6,7,5,8,9)
arithmetic.mean(dataSMMI3$V3)
total<-arithmetic
plot(x = dataSMMI3$V1, y = dataSMMI3$V3,xlab = "Observaciones",ylab = "SSIM", main = "Dispercion de los Datos de la Metrica SSIM",col ="RED" ,pch = 19)
abline(h=mean(dataSMMI3$V3),lty=30,col=86)
legend(800, 0.4, legend=c("Línea 1", "Línea 2"),col=c("forestgreen", "blue"), lty=1:2, cex=0.8)
       


arrange(dataPSNR$V2,a = as.double(dataPSNR$V3))

