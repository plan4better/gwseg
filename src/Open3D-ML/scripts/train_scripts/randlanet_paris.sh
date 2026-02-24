#!/bin/bash
#SBATCH -p gpu 
#SBATCH -c 4 
#SBATCH --gres=gpu:1 

if [ "$#" -ne 2 ]; then
    echo "Please, provide the training framework: torch/tf and dataset path"
    exit 1
fi

cd ../..
python scripts/run_pipeline.py $1 -c ml3d/configs/randlanet_parislille3d.yml \
--dataset_path $2


# python scripts/run_pipeline.py torch -c ml3d/configs/randlanet_parislille3d_50c.yml --dataset_path ./data/Paris_Lille3D
# /usr/bin/env /Users/majkshkurti/Developer/P4B/sidewalk-detection/.venv/bin/python /Users/majkshkurti/Developer/P4B/sidewalk-detection/src/Open3D-ML/scripts/run_pipeline.py torch -c /Users/majkshkurti/Developer/P4B/sidewalk-detection/src/Open3D-ML/ml3d/configs/randlanet_parislille3d.yml --dataset.dataset_path /Users/majkshkurti/Developer/P4B/sidewalk-detection/src/Open3D-ML/data/Paris_Lille3D --pipeline SemanticSegmentation --dataset.use_cache True
# /usr/bin/env /Users/majkshkurti/Developer/P4B/sidewalk-detection/.venv/bin/python /Users/majkshkurti/Developer/P4B/sidewalk-detection/src/Open3D-ML/scripts/run_pipeline.py torch -c /Users/majkshkurti/Developer/P4B/sidewalk-detection/src/Open3D-ML/ml3d/configs/randlanet_parislille3d_50c.yml --dataset.dataset_path /Users/majkshkurti/Developer/P4B/sidewalk-detection/src/Open3D-ML/data/Paris_Lille3D --pipeline SemanticSegmentation --dataset.use_cache True
# /usr/bin/env /Users/majkshkurti/Developer/P4B/sidewalk-detection/.venv/bin/python /Users/majkshkurti/Developer/P4B/sidewalk-detection/src/Open3D-ML/scripts/run_pipeline.py torch -c /Users/majkshkurti/Developer/P4B/sidewalk-detection/src/Open3D-ML/ml3d/configs/randlanet_parislille3d.yml --dataset.dataset_path /Users/majkshkurti/Developer/P4B/sidewalk-detection/src/Open3D-ML/data/Paris_Lille3D --pipeline SemanticSegmentation --dataset.use_cache True
