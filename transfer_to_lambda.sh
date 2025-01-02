
ip_address="150.136.47.139"
local_path="/Users/alishehper/work/feature_universality"
remote_path="/home/ubuntu/storage/feature_universality"
rsync -avz -e ssh $local_path/llm_checkpoints/8-512 ubuntu@$ip_address:$remote_path/llm_checkpoints/