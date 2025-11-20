# Automated Documentation Report

Generated: 2025-10-28 17:32:44

## ga_sbx::ga_sbx
**Config**:
```json
{
  "name": "ga_sbx_dim68_pop40_gens40_procW18",
  "train_scenarios": [
    {
      "name": "training2",
      "num_asteroids": null,
      "asteroid_states": [
        {
          "position": [
            100,
            100
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            200
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            300
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            400
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            500
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            600
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            700
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        }
      ],
      "ship_states": [
        {
          "position": [
            400,
            400
          ],
          "angle": 90,
          "lives": 5,
          "team": 1
        }
      ],
      "map_size": [
        1000,
        800
      ],
      "time_limit": 60.0,
      "ammo_limit_multiplier": 0.0,
      "stop_if_no_ammo": false
    }
  ],
  "eval_scenarios": [
    {
      "name": "training2",
      "num_asteroids": null,
      "asteroid_states": [
        {
          "position": [
            100,
            100
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            200
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            300
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            400
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            500
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            600
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            700
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        }
      ],
      "ship_states": [
        {
          "position": [
            400,
            400
          ],
          "angle": 90,
          "lives": 5,
          "team": 1
        }
      ],
      "map_size": [
        1000,
        800
      ],
      "time_limit": 60.0,
      "ammo_limit_multiplier": 0.0,
      "stop_if_no_ammo": false
    }
  ],
  "dim": 68,
  "optimizer": "ga_sbx",
  "pop": 40,
  "gens": 40,
  "seeds": 3,
  "episodes": 3,
  "workers": 18,
  "eval_timeout": null,
  "train_max_hours": 1,
  "frozen_random_per_gen": false,
  "frozen_maps_per_gen": 3,
  "frozen_num_asteroids": 12,
  "frozen_map_w": 1200,
  "frozen_map_h": 900
}
```

**Training (per-seed)**:
- seed 0: best=353.4555984555985
- seed 1: best=325.2683982683983
- seed 2: best=372.864768683274

**Evaluation per seed**:
- seed 0: fitness_mean=353.456±0.000, score_mean=nan±nan, acc_mean=0.985±0.000, hits_mean=255.000±0.000, deaths_mean=0.000±0.000
- seed 1: fitness_mean=325.268±0.000, score_mean=nan±nan, acc_mean=0.983±0.000, hits_mean=227.000±0.000, deaths_mean=0.000±0.000
- seed 2: fitness_mean=372.865±0.000, score_mean=nan±nan, acc_mean=0.979±0.000, hits_mean=275.000±0.000, deaths_mean=0.000±0.000

## ga_uniform::ga_uniform
**Config**:
```json
{
  "name": "ga_uniform_dim68_pop40_gens40_procW18",
  "train_scenarios": [
    {
      "name": "training2",
      "num_asteroids": null,
      "asteroid_states": [
        {
          "position": [
            100,
            100
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            200
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            300
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            400
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            500
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            600
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            700
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        }
      ],
      "ship_states": [
        {
          "position": [
            400,
            400
          ],
          "angle": 90,
          "lives": 5,
          "team": 1
        }
      ],
      "map_size": [
        1000,
        800
      ],
      "time_limit": 60.0,
      "ammo_limit_multiplier": 0.0,
      "stop_if_no_ammo": false
    }
  ],
  "eval_scenarios": [
    {
      "name": "training2",
      "num_asteroids": null,
      "asteroid_states": [
        {
          "position": [
            100,
            100
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            200
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            300
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            400
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            500
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            600
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            700
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        }
      ],
      "ship_states": [
        {
          "position": [
            400,
            400
          ],
          "angle": 90,
          "lives": 5,
          "team": 1
        }
      ],
      "map_size": [
        1000,
        800
      ],
      "time_limit": 60.0,
      "ammo_limit_multiplier": 0.0,
      "stop_if_no_ammo": false
    }
  ],
  "dim": 68,
  "optimizer": "ga_uniform",
  "pop": 40,
  "gens": 40,
  "seeds": 3,
  "episodes": 3,
  "workers": 18,
  "eval_timeout": null,
  "train_max_hours": 1,
  "frozen_random_per_gen": false,
  "frozen_maps_per_gen": 3,
  "frozen_num_asteroids": 12,
  "frozen_map_w": 1200,
  "frozen_map_h": 900
}
```

**Training (per-seed)**:
- seed 0: best=321.11504424778764
- seed 1: best=308.14814814814815
- seed 2: best=337.41434262948206

**Evaluation per seed**:
- seed 0: fitness_mean=321.115±0.000, score_mean=nan±nan, acc_mean=0.991±0.000, hits_mean=226.000±0.000, deaths_mean=2.000±0.000
- seed 1: fitness_mean=308.148±0.000, score_mean=nan±nan, acc_mean=0.981±0.000, hits_mean=214.000±0.000, deaths_mean=2.000±0.000
- seed 2: fitness_mean=337.414±0.000, score_mean=nan±nan, acc_mean=0.964±0.000, hits_mean=243.000±0.000, deaths_mean=1.000±0.000

## de::de
**Config**:
```json
{
  "name": "de_dim68_pop40_gens40_procW18",
  "train_scenarios": [
    {
      "name": "training2",
      "num_asteroids": null,
      "asteroid_states": [
        {
          "position": [
            100,
            100
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            200
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            300
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            400
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            500
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            600
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            700
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        }
      ],
      "ship_states": [
        {
          "position": [
            400,
            400
          ],
          "angle": 90,
          "lives": 5,
          "team": 1
        }
      ],
      "map_size": [
        1000,
        800
      ],
      "time_limit": 60.0,
      "ammo_limit_multiplier": 0.0,
      "stop_if_no_ammo": false
    }
  ],
  "eval_scenarios": [
    {
      "name": "training2",
      "num_asteroids": null,
      "asteroid_states": [
        {
          "position": [
            100,
            100
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            200
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            300
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            400
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            500
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            600
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            700
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        }
      ],
      "ship_states": [
        {
          "position": [
            400,
            400
          ],
          "angle": 90,
          "lives": 5,
          "team": 1
        }
      ],
      "map_size": [
        1000,
        800
      ],
      "time_limit": 60.0,
      "ammo_limit_multiplier": 0.0,
      "stop_if_no_ammo": false
    }
  ],
  "dim": 68,
  "optimizer": "de",
  "pop": 40,
  "gens": 40,
  "seeds": 3,
  "episodes": 3,
  "workers": 18,
  "eval_timeout": null,
  "train_max_hours": 1,
  "frozen_random_per_gen": false,
  "frozen_maps_per_gen": 3,
  "frozen_num_asteroids": 12,
  "frozen_map_w": 1200,
  "frozen_map_h": 900
}
```

**Training (per-seed)**:
- seed 0: best=313.33333333333337
- seed 1: best=329.3050847457627
- seed 2: best=340.99196787148594

**Evaluation per seed**:
- seed 0: fitness_mean=313.333±0.000, score_mean=nan±nan, acc_mean=0.973±0.000, hits_mean=222.000±0.000, deaths_mean=3.000±0.000
- seed 1: fitness_mean=329.305±0.000, score_mean=nan±nan, acc_mean=0.983±0.000, hits_mean=233.000±0.000, deaths_mean=1.000±0.000
- seed 2: fitness_mean=340.992±0.000, score_mean=nan±nan, acc_mean=0.980±0.000, hits_mean=245.000±0.000, deaths_mean=1.000±0.000

## cma::cma
**Config**:
```json
{
  "name": "cma_dim68_pop40_gens40_procW18",
  "train_scenarios": [
    {
      "name": "training2",
      "num_asteroids": null,
      "asteroid_states": [
        {
          "position": [
            100,
            100
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            200
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            300
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            400
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            500
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            600
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            700
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        }
      ],
      "ship_states": [
        {
          "position": [
            400,
            400
          ],
          "angle": 90,
          "lives": 5,
          "team": 1
        }
      ],
      "map_size": [
        1000,
        800
      ],
      "time_limit": 60.0,
      "ammo_limit_multiplier": 0.0,
      "stop_if_no_ammo": false
    }
  ],
  "eval_scenarios": [
    {
      "name": "training2",
      "num_asteroids": null,
      "asteroid_states": [
        {
          "position": [
            100,
            100
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            200
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            300
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            400
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            500
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            600
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            700
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        }
      ],
      "ship_states": [
        {
          "position": [
            400,
            400
          ],
          "angle": 90,
          "lives": 5,
          "team": 1
        }
      ],
      "map_size": [
        1000,
        800
      ],
      "time_limit": 60.0,
      "ammo_limit_multiplier": 0.0,
      "stop_if_no_ammo": false
    }
  ],
  "dim": 68,
  "optimizer": "cma",
  "pop": 40,
  "gens": 40,
  "seeds": 3,
  "episodes": 3,
  "workers": 18,
  "eval_timeout": null,
  "train_max_hours": 1,
  "frozen_random_per_gen": false,
  "frozen_maps_per_gen": 3,
  "frozen_num_asteroids": 12,
  "frozen_map_w": 1200,
  "frozen_map_h": 900
}
```

**Training (per-seed)**:
- seed 0: best=181.80219780219778
- seed 1: best=190.9795918367347
- seed 2: best=190.0

**Evaluation per seed**:
- seed 0: fitness_mean=181.802±0.000, score_mean=nan±nan, acc_mean=0.978±0.000, hits_mean=94.000±0.000, deaths_mean=5.000±0.000
- seed 1: fitness_mean=190.980±0.000, score_mean=nan±nan, acc_mean=0.990±0.000, hits_mean=102.000±0.000, deaths_mean=5.000±0.000
- seed 2: fitness_mean=190.000±0.000, score_mean=nan±nan, acc_mean=1.000±0.000, hits_mean=100.000±0.000, deaths_mean=5.000±0.000
