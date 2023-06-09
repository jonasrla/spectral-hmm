cp .tex/* . > /dev/null 2>&1
biber template
docker run -ti \
  -v miktex:/miktex/.miktex \
  -v `pwd`:/miktex/work \
  --rm \
  miktex/miktex \
  pdflatex $1.tex

open $1.pdf
mkdir .tex > /dev/null 2>&1
mv *.aux .tex
mv *.bbl .tex
mv *.log .tex
mv *.out .tex
mv *.xml .tex
mv *.blg .tex

