


# Run these commands from my Macbook. DO NOT RUN THIS BASH SCRIPT AS IT IS.
#rsync -avz -e ssh mas1107@amarel.rutgers.edu:/scratch/mas1107/feature_universality/llm_checkpoints/8-768/ckpt_final.pt /Users/alishehper/Downloads/8-768/
#rsync -avz -e ssh /Users/alishehper/Downloads/8-768/ckpt_final.pt zara-ss@apollo:/home/zara-ss/Documents/feature_universality/llm_checkpoints/8-768/

rsync -avz -e ssh mas1107@amarel.rutgers.edu:/scratch/mas1107/feature_universality/sae_checkpoints /Users/alishehper/work/feature_universality
rsync -avz -e ssh /Users/alishehper/work/feature_universality/sae_checkpoints zara-ss@apollo:/home/zara-ss/Documents/feature_universality/