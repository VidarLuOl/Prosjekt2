# Prosjekt2
Prosjekt2 Fys-stk4155

[Prosjekt2](https://compphysics.github.io/MachineLearning/doc/Projects/2021/Project2/html/Project2-bs.html)

[Overleaf](https://www.overleaf.com/project/6167f4a28b15e3ffa0aab71c)

Steg A:
 - Erstatt den inverse matrisen i OLS/Ridge fra innlevering 1 med SGD (https://compphysics.github.io/MachineLearning/doc/pub/week40/html/week40.html). Enten "momentum SGD optionality" eller and SGD varianter
 - Analyser resultatet for OLS og Ridge som en funksjon av learning rates, antall "mini-batches" og "epochs", i tillegg til hvordan å skalere dataen påvirker.
 - Du kan sammenligne med Scikit-Learn's ulike SGD. Diskuter resultatet.
 - For Ridge så må vi nå studere resultatet som en fnksjon av hyper-parameteren lambda og "learning rate" psi? DIskuter

Steg B:
 - Skriv en Feed Forward Neural Network kode som implimenterer "Back propagation algorithm" diskuter i (https://compphysics.github.io/MachineLearning/doc/pub/week41/html/week41.html).
 - Vi fokuserer først på et regressionsproblem først og studere enten Franke funksjonen eller terreng dataen fra prosjekt 1. Diskuter igjen valget av cost funksjon.
 - Lag en FFNN kode for regressjon med et fleksibelt antall "hidden layers" og noder ved å bruke Sigmoid funksjon som en aktiverings funksjon for "hidden layers". Initialiser vektene ved å bruke en normal distribusjon. 
 - Hvordan vil me initialisere biasene? Og hvilken aktiverings funksjon velger en for den siste output layer.
 - Tren netverket og sammenlign med resultatene fra OLS og Ridge regression fra prosjekt 1.  Sammenlign med en kode hvor en bruker Scikit-Learn eller tensorflow/keras.
 - Kommenter resultatet, gi en kritisk diskusjon av resultatet med Linear Regression koden og Neural Network koden. Sammenlign kodene med de fra prosjekt 1. 
 - Lag en analyse for "regularization parameteres" og "learning raten" ansatt for å finna optimal MSE og R2 score.

Steg C:
 - Gjør detta senere