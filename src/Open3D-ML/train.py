import open3d.ml.torch as ml3d
from open3d.ml.torch.pipelines import SemanticSegmentation
import open3d.ml as _ml3d

# use a cache for storing the results of the preprocessing (default path is './logs/cache')
# cfg_file = 'src/Open3D-ML/ml3d/configs/randlanet_parislille3d_50c.yml'
cfg_file = 'src/Open3D-ML/ml3d/configs/randlanet_parislille3d.yml'
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
model = ml3d.models.RandLANet(**cfg.model)

dataset = ml3d.datasets.ParisLille3D(dataset_path='src/Open3D-ML/data/Paris_Lille3D', use_cache=True)


pipeline = SemanticSegmentation(model=model, dataset=dataset, **cfg.pipeline, num_workers=0)

# prints training progress in the console.
pipeline.run_train()