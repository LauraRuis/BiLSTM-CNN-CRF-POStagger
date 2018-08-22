#VERSION="3_3_0"
VERSION="3_5_0"
PARSER="/home/joost/git/stanford-parser-${VERSION}"

base="/scratch/joost/data/treebank3/treebank_3/parsed/mrg/wsj"
outputDir="wsj-conllx-${VERSION}"

if [ ! -d ${outputDir} ]; then
  mkdir ${outputDir}
fi
for dir in ${base}/*
do
  resultDir="${outputDir}/`basename ${dir}`"
  if [ ! -d ${resultDir} ]; then
    mkdir ${resultDir}
  fi
  for f in ${dir}/*.mrg
  do
    outputFile="${resultDir}/`basename "${f}" .mrg`.conllx"
    java -cp "$PARSER/*" -mx1g edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -keepPunct -conllx -treeFile "${f}" > "${outputFile}"
    echo "Processed ${f}, output in ${outputFile}"
  done
done

