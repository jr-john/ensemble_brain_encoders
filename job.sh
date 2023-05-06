#!/bin/bash
#SBATCH -A research
#SBATCH -c 9
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 2G
#SBATCH --time 4-00:00:00
#SBATCH --output job-logs/csai.log
#SBATCH --mail-user jerrin.thomas@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name csai

echo "Setting Up Environment"
cd /scratch
[ -d "jerrin.thomas" ] || mkdir jerrin.thomas
chmod 700 jerrin.thomas
cd jerrin.thomas
rm -rf ./*

cp -r ~/projects/csai/* .
scp -r jerrin.thomas@ada:/share1/jerrin.thomas/csai/* .

# for task in bert coref ner nli paraphrase qa sa srl ss sum wsd
# do
#     echo "Running Task: $task"
#     python3 feature_extractor.py $task
#     python3 encoder.py $task
# done
python3 ensemble.py
python3 ensemble_evaluation.py
python3 analysis.py

mkdir new
cp ensemble_wt_avg.pkl new
cp results/results_ensemble_wt_avg.pkl new
cp results/2v2.png new
cp results/pear.png new

scp -r new jerrin.thomas@ada:/share1/jerrin.thomas/